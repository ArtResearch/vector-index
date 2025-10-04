# Metadata Search Server

This document provides an overview of the `searchUSearchIndex_server`, its capabilities, and how to use it.

## Overview

The `searchUSearchIndex_server` is a high-performance C++ server designed for vector similarity search. It uses the `usearch` library for efficient indexing and searching of high-dimensional vectors (embeddings). The server exposes an HTTP API to perform searches and can be integrated into larger systems, particularly those using SPARQL for querying.

A key feature of this server is its ability to handle multiple search models simultaneously and join the results from these models based on specified metadata columns.

## Features

-   **High-Performance Search:** Leverages `usearch` for fast and memory-efficient vector similarity searches.
-   **Multi-Model Support:** Can load and serve multiple `usearch` indexes, each with its own associated metadata.
-   **Dynamic Metadata Handling:** Supports metadata files (CSV) with an arbitrary number of columns. The first row of the CSV is expected to be the header.
-   **SPARQL Endpoint:** Provides a SPARQL endpoint that translates custom search predicates into vector search operations.
-   **Multi-Model Joins:** Supports joining results from multiple models on arbitrary metadata columns using the `joinOn` parameter.

## Running the Server

To run the server, you need to provide arguments for each model you want to load. The number of arguments for `--model`, `--index`, `--metadata`, and `--embedding_socket` must be the same.

```bash
./searchUSearchIndex_server \\
    --model model_name_1 --index /path/to/index_1.usearch --metadata /path/to/metadata_1.csv --embedding_socket /path/to/socket_1.sock \\
    --model model_name_2 --index /path/to/index_2.usearch --metadata /path/to/metadata_2.csv --embedding_socket /path/to/socket_2.sock \\
    --port 8545 \\
    --threads 4
```

### Command-Line Arguments

-   `--model, -n`: The name of the model.
-   `--index, -i`: Path to the `usearch` index file.
-   `--metadata, -m`: Path to the corresponding metadata CSV file. The CSV must contain a header row.
-   `--embedding_socket, -e`: Path to the Unix socket for the embedding service.
-   `--port, -p`: The port for the search server (default: 8545).
-   `--threads, -j`: Number of threads to use (default: hardware concurrency).
-   `--sensitivity-defaults`: Default limits for sensitivity levels (e.g., "100,1000,5000").
-   `--sensitivity-factors`: Standard deviation multipliers for sensitivity cutoffs (e.g., "0.25,-1.0,-2.0").

## API Endpoints

### 1. `/search` (GET)

A simple endpoint for direct similarity search.

**Parameters:**

-   `model`: (Required) The name of the model to search against.
-   `text`: The text to search for.
-   `id`: The ID (key) to search for.
-   `count`: The number of results to return (default: 100).

### 2. `/sparql` (POST)

A powerful endpoint that integrates vector search into SPARQL queries.

The body of the POST request should be a SPARQL query containing a `SERVICE` clause that uses custom predicates for the search.

#### Custom SPARQL Predicates

The search is defined within a `SERVICE emb:search { ... }` block.

-   `?uri emb:data "search_term"`: The data to search for (e.g., text, URL).
-   `?uri emb:request_type "type"`: The type of search (`text`, `uri`, `url`, `image`).
-   `?uri emb:model "model1,model2,..."`: A comma-separated list of models to query.
-   `?uri emb:joinOn "col1,col2"`: (Optional) A comma-separated list of metadata columns to join the results on. Defaults to "uri".
-   `?uri emb:returnValues "col1,col2"`: (Optional) A comma-separated list of metadata columns to use for the returned URI. Defaults to "uri".
-   `?uri emb:score ?similarity`: Binds the similarity score to the `?similarity` variable.
-   `?uri emb:matchedModels ?matchedModel`: Binds the name(s) of the matched model(s) to the `?matchedModel` variable.
-   `?uri emb:maxScore ?aiSimilarityMaxScore`: Binds the maximum similarity score for the query to the `?aiSimilarityMaxScore` variable.
-   `?uri emb:sensitivity "level"`: (Optional) Sets a dynamic result cutoff. Accepted values are `precise`, `balanced`, and `exploratory`. When used, the server fetches up to 10,000 results, analyzes the similarity score decay, and intelligently trims the result set. If this parameter is present, the `LIMIT` clause is ignored. The logic for this trimming is based on the mean and standard deviation of the final merged scores, and can be fine-tuned with the `--sensitivity-factors` argument. For a detailed explanation of the algorithm, see `metadata/src/cutoffideas.md`.

#### Multi-Model Search with `joinOn` and `returnValues`

The `joinOn` and `returnValues` parameters are crucial for multi-model searches. They allow you to specify which columns should be used to join the results from different models and which column should be used for the final returned URI.

-   If one column is provided (e.g., `emb:joinOn "uri"`), it will be used for all models.
-   If multiple columns are provided (e.g., `emb:joinOn "work,uri"`), the first column (`work`) is used for the first model in the `emb:model` list, and the second column (`uri`) is used for the second model. If there are more models than columns, the last specified column is used for the remaining models.

The `returnValues` parameter follows the same logic as `joinOn`. It determines which column's value is returned as the `?uri` in the final result. This is particularly useful when you are joining on a common identifier but want to return a different value from the highest-scoring result.

When results are joined, the entry with the highest score for a given join key is kept. The `?uri` returned will be from the `returnValues` column of the highest-scoring entry.

If a metadata column contains comma-separated values (e.g., `"value1,value2"`), each value is treated as a separate key for the purpose of joining.

**Example Query:**

This query searches for "flowers" in two models (`clip-vit-large-patch14-336` and `qwen3-8b`). It joins the results from the first model on the `work` column with the results from the second model on the `uri` column. The final `?uri` returned will be from the `work` column of the first model and the `uri` column of the second model, depending on which has the higher score.

```sparql
PREFIX emb: <https://artresearch.net/embeddings/>
SELECT ?uri ?similarity ?matchedModel
WHERE {
  SERVICE <http://usearch-search.private:8545/sparql> {
    SELECT * {
      SERVICE emb:search {
        ?uri emb:data "flowers" .
        ?uri emb:request_type "text" .
        ?uri emb:model "clip-vit-large-patch14-336,qwen3-8b" .
        ?uri emb:joinOn "work,uri" .
        ?uri emb:returnValues "work,uri" .
        ?uri emb:score ?similarity .
        ?uri emb:matechedModels ?matchedModel . 
        ?uri emb:maxScore ?aiSimilarityMaxScore .
      }
    } LIMIT 10000
  }
}
ORDER BY DESC(?similarity)
