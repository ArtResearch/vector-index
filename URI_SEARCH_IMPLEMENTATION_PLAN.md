# Specification and Implementation Plan: URI-Based Similarity Search (Revised)

This revision incorporates concrete improvements identified during code review, adds robustness and performance considerations, and clarifies API semantics. It also introduces optional OpenMP-driven parallel expansion/search to speed up URI-based batch queries.

## 1. Objective

Extend the search service to support a new `request_type` `"uri"`, enabling:

1. Simple Similarity Search: Find items similar to a single, specified identifier (typically a URI) within one or more models.
2. Advanced Cross-Modal Join Search: Use a given identifier (which could be a URI, a "work" ID, or any other metadata field) to find a set of related URIs across different models and then perform a similarity search for all of them, combining the results.

This feature reuses the existing result aggregation and scoring logic, but adds a new entry point for initiating searches based on identifiers instead of raw text/image inputs.

## 2. SPARQL Interface and Query Semantics

The new functionality is controlled via `request_type`, `joinOn`, and `returnValues` parameters in the SPARQL query.

- request_type must be `"uri"` to activate the URI path.
- emb:data is the input identifier (e.g., `http://example.com/work/123`).
- emb:joinOn controls which metadata column to match the input identifier against.
- emb:returnValues controls the column used for the final grouping key.

Key rules and improvements:
- Default joinOn: If `joinOn` is omitted, it MUST default to `"uri"` (not to `returnValues`). This ensures predictable semantics.
- Token trimming: Comma-separated lists for `model`, `joinOn`, and `returnValues` must be split and TRIMMED to avoid header-name mismatches due to whitespace.
- Positional mapping: If multiple models and joinOn/returnValues are provided, they map positionally (e.g., `model: m1,m2`, `joinOn: c1,c2` maps `c1`→`m1`, `c2`→`m2`). If fewer columns than models are provided, the last one applies to the remainder.
- limit: No changes to existing `limit` semantics. The current handling of result trimming/limits remains as-is.

### Example Queries

Example 1: Find similar works
Find all items belonging to the same work as `http://work`, search for all of them, and return the most similar works.
```sparql
SERVICE <https://artresearch.net/embeddings/search> {
  ?uri <https://artresearch.net/embeddings/data> "http://work";
    <https://artresearch.net/embeddings/request_type> "uri";
    <https://artresearch.net/embeddings/model> "siglip2_model";
    <https://artresearch.net/embeddings/joinOn> "work";
    <https://artresearch.net/embeddings/returnValues> "work";
}
```

Example 2: Cross-model search
Use `http://work` as a URI in the `qwen` model and as a `work` ID in the `siglip` model to find an initial set of URIs, then search and join the results.
```sparql
SERVICE <https://artresearch.net/embeddings/search> {
  ?uri <https://artresearch.net/embeddings/data> "http://work";
    <https://artresearch.net/embeddings/request_type> "uri";
    <https://artresearch.net/embeddings/model> "qwen_model,siglip_model";
    <https://artresearch.net/embeddings/joinOn> "uri,work";
    <https://artresearch.net/embeddings/returnValues> "uri,work";
}
```

## 3. Data Structures and Indexing

To efficiently expand identifiers into sets of USearch keys, we need fast metadata lookups. We will support a selective metadata indexing strategy to balance performance and memory footprint.

### ModelData Additions (src/index/search/model.hpp)

Add the following fields:

```cpp
struct ModelData {
    // Existing:
    index_t index;
    std::vector<MetadataRow> metadata_table;
    std::vector<std::string> metadata_header;
    std::unordered_map<std::string, size_t> metadata_header_map;
    std::unordered_map<std::string, usearch_key_t> uri_to_key;
    std::string embedding_socket_path;
    std::unique_ptr<fifo_cache<std::string, std::vector<float>>> embedding_cache;
    std::unique_ptr<fifo_cache<std::string, std::vector<SiftMatchResult>>> sift_cache;
    std::unordered_map<std::string, std::string> uri_to_file_map;

    // New (selective indexing of metadata columns):
    // column -> (value -> vector<key>)
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::vector<usearch_key_t>>
    > metadata_value_to_keys;

    // Optional: resolve metadata row reliably when key != row index (safety)
    std::unordered_map<usearch_key_t, size_t> key_to_row_index;
};
```

Notes:
- `uri_to_key` remains the fast path for direct URI lookups.
- `metadata_value_to_keys` is built only for selected columns (e.g., `uri`, `work`), not for every column by default, to control memory.
- `key_to_row_index` avoids assuming `key == row index`. If index/CSV order matches exactly, this can remain a no-op; otherwise populate it defensively.

## 4. Metadata Loading (src/index/search/metadata.cpp)

Populate the new maps while reading CSVs:

- Build `metadata_header` and `metadata_header_map` (existing).
- Populate `uri_to_key` as today.
- Populate `metadata_value_to_keys` only for indexed columns. Indexed columns are configured via a CLI option (see below).
- Handle multi-valued cells for indexed columns: if a cell contains comma-separated values (e.g., `"w1,w2"`), split, trim each token, and index all tokens to the same key.
- Normalize/trim values for indexed columns to avoid mismatches due to extra whitespace.
- Optionally populate `key_to_row_index[key] = current_row_index`.

CLI addition (src/index/app/main.cpp):
- Introduce `--indexed-metadata-columns` (comma-separated). Example: `--indexed-metadata-columns uri,work`.
  - If omitted, default to `"uri"` (always build for `uri`) to match default `joinOn`.
  - Parse once per model and pass list into the metadata loader so only those columns are indexed.

Memory/Performance Notes:
- Selective indexing avoids duplicating the entire metadata table in memory.
- Lazy indexing is also possible: build `metadata_value_to_keys[col]` on first query that references `col`, then cache it.

## 5. Service Logic (src/index/search/service.cpp)

### Request Routing

At the top of `Service::search`, branch on `query.request_type`:

- `"uri"`: new path described below.
- `"url"` / `"image"`: existing image path.
- Otherwise: treated as text (existing path).

### URI Expansion and Batch Search

High-level algorithm:
1. Determine the `joinOn` column for each model (positional mapping, default `"uri"`).
2. Expand the input identifier into a set of USearch keys per model:
   - If `joinOn == "uri"`, use `uri_to_key` fast path (exact match).
   - Otherwise, look up `metadata_value_to_keys[joinOn][query.data]` to get vector<key_t>.
   - For multi-valued indexed columns, expansion works transparently because we indexed tokens individually.
   - Deduplicate keys per model.

3. Cap expansion per model to avoid combinatorial blow-up (configurable, e.g., `--uri-expansion-cap` or sensitivity-aware heuristic).

4. Compute seed exclusion sets for downstream filtering:
   - seed_query_keys (all expanded keys for the current model).
   - seed_uri_set (URIs of all seed rows resolved from metadata).
   - seed_group_set_return (if `returnValues != "uri"`, a set of group_key values corresponding to all seed rows, e.g., set of `work` IDs when returning by `work`).
   - These sets are used to filter both per-match results and aggregated results (see “Filtering semantics” below).

5. For each key, retrieve its stored vector and run a KNN search:
   - `std::vector<float> q(dim); size_t n = model_data.index.get(key, q.data());`
   - assert `n >= 1`; if multi-vectors-per-key ever enabled, iterate them.
   - `auto result = model_data.index.search(q.data(), search_limit, index_t::any_thread(), query.exact_search);`

6. Convert each match to `IntermediateResult` using metadata:
   - Resolve metadata row via `key_to_row_index` if present, otherwise by direct indexing if the invariant `key == row` holds.
   - Populate `group_key` (from returnValues column), `join_key` (from joinOn column), `score` (normalized similarity), `original_uri`, `model_name`.

7. Filtering semantics (two levels):
   - Per-match filtering:
     - Remove self-matches (matched key ∈ seed_query_keys).
     - Remove any match whose `original_uri` ∈ seed_uri_set (e.g., when searching a work with multiple photos, exclude all those photos).
   - Aggregation-time filtering:
     - If `returnValues == "uri"`, aggregated results must still exclude any record whose `original_uri` ∈ seed_uri_set.
     - If `returnValues != "uri"`, exclude any aggregated group whose `group_key` ∈ seed_group_set_return (e.g., when returning by `work`, exclude the seed work itself).

8. Aggregate and score using the existing pipeline (unchanged), including:
   - Keep the existing “max score per (group_key, model_name)” behavior.
   - Preserve any existing “maxScoreUri” logic in the codebase. The URI path feeds the same `IntermediateResult` structures, so the per-URI max logic remains effective without changes.

9. Apply sensitivity trimming (unchanged).

10. Do not change `limit` behavior:
   - The current semantics for limiting/trimming results remain unchanged. No new enforcement is added in this plan.

### Parallelization with OpenMP (optional)

To speed up expansion searches, parallelize the per-key search within each model using OpenMP:

- Guards:
  - Use `#ifdef _OPENMP` to enable at compile-time.
  - Ensure USearch’s dense index has enough internal threads (see “Threading considerations” below).
- Pattern:
  ```cpp
  // Pseudocode inside the "uri" branch per model
  const auto& keys = keys_to_search_for_model;
  std::vector<IntermediateResult> partial_results;
  partial_results.reserve(keys.size() * expected_k);

  #pragma omp parallel for schedule(dynamic) if (keys.size() > 1)
  for (size_t i = 0; i < keys.size(); ++i) {
      usearch_key_t key = keys[i];

      // 1) Retrieve vector
      std::vector<float> q(dim);
      if (model_data.index.get(key, q.data()) < 1) continue;

      // 2) Search (pick any_thread; the wrapper manages thread slots)
      auto knn = model_data.index.search(q.data(), search_limit, index_t::any_thread(), query.exact_search);

      // 3) Dump and convert
      std::vector<usearch_key_t> found_keys(knn.size());
      std::vector<distance_t> found_distances(knn.size());
      knn.dump_to(found_keys.data(), found_distances.data());

      // Normalize scores by max similarity (1 - dist)
      float max_sim = 0.f;
      for (auto d : found_distances) max_sim = std::max(max_sim, 1.0f - d);
      if (max_sim <= 0.f) continue;

      std::vector<IntermediateResult> local_buffer;
      local_buffer.reserve(knn.size());
      for (size_t j = 0; j < knn.size(); ++j) {
          // Per-match filtering
          if (seed_query_keys.count(found_keys[j])) continue;

          size_t row_idx = resolve_row(found_keys[j], model_data); // via key_to_row_index if present
          const auto& meta = model_data.metadata_table[row_idx];
          const std::string& candidate_uri = meta[uri_column_index];
          if (seed_uri_set.count(candidate_uri)) continue;

          IntermediateResult res;
          res.group_key = meta[return_column_index];
          res.join_key = meta[join_column_index];
          res.score = (1.0f - found_distances[j]) / max_sim;
          res.original_uri = candidate_uri;
          res.model_name = model_name;
          local_buffer.push_back(std::move(res));
      }

      // 4) Thread-safe merge
      #pragma omp critical
      {
          partial_results.insert(partial_results.end(), local_buffer.begin(), local_buffer.end());
      }
  }

  // Aggregation-time filtering:
  // - If returnValues == "uri": drop entries where original_uri ∈ seed_uri_set.
  // - Else: drop entries whose group_key ∈ seed_group_set_return.
  ```
- Threading considerations:
  - `index_dense_gt::search` owns a thread lock internally (`index_t::any_thread()`), backed by an “available threads” ring. If you parallelize with OpenMP, don’t spawn more parallel tasks than `index.limits().threads()` for that model, otherwise you may exhaust the ring and hit assertions. You can bound `omp_set_num_threads(std::min(omp_get_max_threads(), (int)model_data.index.limits().threads()))`, or drop OpenMP to sequential when `keys.size()` is small.
  - Treat `partial_results` as thread-local and merge under a small critical section.

If OpenMP isn’t desired, a std::thread pool or serial execution is fine. Keep the expansion cap to avoid wasting work.

### Additional Robustness

- Column validation: If `joinOn` or `returnValues` column name doesn’t exist in `metadata_header_map`, return an explicit error or warning with the offending model/column.
- Normalization: For indexed metadata values, trim both at load time and query time, and document case-sensitivity. For URIs, match exact/case-sensitive. For IDs like `work`, choose a consistent normalization strategy (e.g., trim and case-sensitive unless otherwise specified).
- Compatibility with existing “maxScoreUri” logic:
  - No changes required. The URI path produces the same `IntermediateResult` tuples with `original_uri` populated; the existing per-URI maximum score logic continues to apply unchanged.

## 6. Parser Changes (src/index/sparql/parser.cpp)

- Default `joinOn` to `"uri"` if omitted. Do NOT fall back to `returnValues`.
- Trim tokens after splitting for:
  - `model`
  - `joinOn`
  - `returnValues`
- Keep `limit`, `exact`, and `sensitivity` logic unchanged (no changes to how `limit` is applied downstream).

## 7. Areas Not Requiring Changes

- Response formatter (`src/index/sparql/response_formatter.cpp`): Unchanged.
- Embedding client: Unused in `"uri"` path, since we fetch in-index vectors.
- SIFT reranking: Not involved in URI path (only used for image-based requests).

## 8. Configuration Additions (Optional)

- `--indexed-metadata-columns`: Comma-separated list of columns to index in `metadata_value_to_keys`. If omitted, default is `"uri"`. Pass to the metadata loader.
- `--uri-expansion-cap`: Upper bound on the number of expanded keys per model to guard against explosion. Optionally make it sensitivity-aware.

## 9. Testing Plan

- Parser unit tests:
  - `joinOn` omitted → defaults to `"uri"`.
  - Whitespace trimming across `model`, `joinOn`, `returnValues`.
- URI path functional tests:
  - Simple: `joinOn` omitted (defaults to `"uri"`), identifier that maps to a single key.
  - Cross-model: different `joinOn` per model, positional mapping works.
  - Multi-valued: `joinOn=work` where cells contain `"w1,w2"`. Ensure both `w1` and `w2` expand.
  - No-match: Fast exit with empty results.
  - Large-match: Respect expansion cap; performance acceptable.
  - Self-match/Exclusion: Seed keys and all seed URIs are excluded; when returning by `work`, the seed work is excluded at aggregation.
  - “maxScoreUri” compatibility: Confirm the highest-scoring result per URI is preserved as before.
  - limit: Verify current behavior remains unchanged.

## 10. Summary of Changes vs. Previous Revision

- Parser semantics kept: `joinOn` defaults to `"uri"`, token trimming added; `limit` behavior unchanged.
- Index API usage: Use `index_dense_gt::get(key, f32*)` + `search(f32*)`.
- Exclusion semantics strengthened:
  - Per-match: exclude self and any candidate URI in the seed set.
  - Aggregation-time: exclude seed `original_uri` when returning `uri`; exclude seed group keys (e.g., seed `work`) when returning by other columns.
- Selective (or lazy) metadata indexing to control memory usage; CLI to configure columns.
- Multi-valued metadata cells properly indexed (split/trim).
- Expansion caps to prevent blow-ups.
- Compatibility with existing “maxScoreUri” logic is preserved.
- Optional OpenMP parallelization for per-key searches with guidance on thread limits.

This plan aligns cleanly with the current architecture, preserves existing limit and “maxScoreUri” behaviors, and addresses correctness, robustness, and performance, including filtering at both match and aggregation stages and efficient parallel execution when desired.
