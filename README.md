# Ignis Search Server

This document describes the functionality and API of the Ignis search server.

## Overview

The Ignis search server is a C++ application that provides a search service over a collection of indexed data. It exposes an HTTP endpoint that accepts SPARQL-like queries to perform searches and retrieve results. The server is highly configurable, allowing you to specify multiple search models, indices, and metadata files.

## How it Works

The server is built around a core search service that manages one or more search models. Each model consists of:

-   A **USearch index** for efficient similarity search.
-   A **metadata file** containing additional information about the indexed items.
-   An **embedding service** that converts search queries into vector embeddings.

When a search query is received, the server performs the following steps:

1.  **Parse the SPARQL Query**: The server parses the incoming SPARQL query to extract search parameters, such as the search term, the desired model, and the number of results to return.
2.  **Generate Embeddings**: The search term (text, image URL, or base64-encoded image) is sent to the configured embedding service, which returns a vector embedding.
3.  **Perform Similarity Search**: The server uses the USearch index to find the most similar items to the query embedding.
4.  **SIFT Reranking (for image queries)**: For image-based searches, an additional reranking step is performed using SIFT feature matching. This refines the search results by comparing the local features of the query image with the candidate images from the initial search, significantly improving accuracy for visually similar items.
5.  **Retrieve Metadata**: The metadata for the top matching items is retrieved from the metadata file.
6.  **Combine and Score**: Results from the initial search and SIFT reranking are combined and assigned a final score.
7.  **Format and Return Response**: The final results are formatted as a SPARQL JSON response and returned to the client.

### SIFT Reranking Logic

For image queries, the server employs a sophisticated SIFT (Scale-Invariant Feature Transform) reranking process to refine the initial search results. This is crucial for identifying true visual matches with high precision, going beyond the capabilities of global image embeddings. The process involves several key stages:

1.  **Feature Extraction with RootSIFT**: Instead of standard SIFT, the server uses RootSIFT descriptors. This is a variant of SIFT that provides improved matching accuracy by applying a Hellinger kernel normalization to the descriptors. Features are extracted from the query image and all candidate images from the initial search.

2.  **High-Quality Feature Matching**: To establish initial correspondences between the query and a candidate image, a robust matching strategy is used:
    *   **Symmetric Matching**: Features are matched in both directions (query-to-candidate and candidate-to-query).
    *   **Lowe's Ratio Test**: This test ensures that matches are distinctive and not ambiguous.
    *   **Mutual Consistency Check**: A match is only kept if it's the best match in both directions, eliminating many false positives.

3.  **Multi-Stage Geometric Verification**: A simple count of matching features is not reliable. Therefore, candidates are filtered through a rigorous geometric verification pipeline to ensure the spatial arrangement of matched features is consistent.
    *   **Hough Transform Pre-filter**: A Hough-style voting process quickly identifies a consensus on the dominant scale and rotation transformation between the images. Candidates without a strong consensus are discarded early.
    *   **Homography with MAGSAC**: For promising candidates, a full geometric model (a homography) is estimated using `MAGSAC`, a highly robust alternative to RANSAC. This step calculates the number of "inliers"—features that fit the geometric model.
    *   **Dynamic Inlier Thresholding**: A candidate is only confirmed as a match if the number of inliers exceeds a dynamic threshold. This threshold considers the number of initial matches, the Hough peak, and the feature density, making the verification robust across different image types and resolutions.

4.  **Final Scoring**: The final SIFT score for a matched image is based on the density of inlier features. This multi-stage filtering process is computationally intensive but extremely effective at confirming genuine visual matches and rejecting geometrically inconsistent false positives.

## API

The server exposes a single HTTP endpoint for performing searches.

-   **Endpoint**: `/sparql`
-   **Method**: `POST`
-   **Content-Type**: `application/sparql-query`

### Request Body

The request body should contain a SPARQL query that specifies the search parameters. The query is parsed using regular expressions to extract the following fields:

-   `emb:data`: The search term (e.g., a string of text or a URL to an image). This field is **required**.
-   `emb:request_type`: The type of the request. This field is **required**. Possible values are:
    -   `text`: The `emb:data` field contains a string of text.
    -   `url`: The `emb:data` field contains a URL to an image.
    -   `image`: The `emb:data` field contains a base64-encoded image.
-   `emb:model`: A comma-separated list of model names to use for the search. This field is **required**.
-   `limit`: The maximum number of results to return. Defaults to `10`.
-   `emb:sensitivity`: The sensitivity of the search. Can be one of `precise`, `balanced`, `exploratory`, or `near-exact`. The `near-exact` sensitivity is specifically designed for image queries and leverages SIFT reranking to find visually identical or near-identical matches.
-   `emb:exact`: A boolean value indicating whether to perform an exact search. Defaults to `false`.
-   `emb:returnValues`: A comma-separated list of metadata fields to return in the response. Defaults to `uri`. When using multiple models, you can provide a corresponding list of return values for each model. If fewer values are provided than models, the last value is used for the remaining models.
-   `emb:joinOn`: A comma-separated list of metadata fields to use for joining results from different models. This is the key to the multi-model search logic. Like `returnValues`, you can specify a different `joinOn` column for each model.
-   `emb:filterBy`: The name of a metadata attribute to filter on. This is an **optional** field.
-   `emb:filterValues`: A comma-separated string of values to filter by. This is an **optional** field and is used in conjunction with `emb:filterBy`.

### Multi-Model Search, Joining, and Aggregation

The server's true power lies in its ability to query multiple models simultaneously and intelligently combine their results. This is controlled by the `emb:model`, `emb:returnValues`, and `emb:joinOn` parameters.

**Logic:**

1.  **Concurrent Search**: When you provide multiple model names (e.g., `"clip,dino"`), the server runs a search against each model in parallel using the same query data.
2.  **Per-Model Aggregation (`returnValues`)**: For each model, you can specify a `returnValues` column. If this column is anything other than the default `uri`, the server will first aggregate the results for that model. It groups the results by the unique values in the `returnValues` column and only keeps the top-scoring result for each unique value. This is useful for preventing near-duplicate results from the same model (e.g., multiple images of the same artwork).
3.  **Cross-Model Joining (`joinOn`)**: After the optional per-model aggregation, the server combines the results from all models using the `joinOn` column. All results that share the same value in their respective `joinOn` column are grouped into a single final result.
4.  **Weighted Scoring**: Each final grouped result is given a `final_weighted_score`. The scoring logic prioritizes high-confidence visual matches:
    *   If any model produced SIFT reranking results for the group, the SIFT results contribute `90%` to the score, and standard embedding-based results contribute `10%`.
    *   If there are no SIFT results, the contributions from each model are weighted equally.

This mechanism allows you to create powerful, federated queries. For example, you can search for an image using a `clip` model (good at semantics) and a `dino` model (good at texture/style), and join the results on an `artwork_id` column to find artworks that are relevant in both semantic and stylistic contexts.

### Example Queries

**1. Simple Text Search**
```sparql
PREFIX emb: <https://artresearch.net/embeddings/>

SELECT ?uri ?title
WHERE {
  SERVICE <https://artresearch.net/embeddings/service> {
    ?work emb:data "a painting of a dog" .
    ?work emb:request_type "text" .
    ?work emb:model "clip" .
    ?work emb:returnValues "uri,title" .
  }
}
LIMIT 10
```

**2. Multi-Model Image Search with Aggregation and Joining**

This query searches for an image using two models, `clip` and `dino`.
- For the `clip` model, it aggregates results by `artwork_id`.
- For the `dino` model, it returns raw `uri` results.
- It then joins the results from both models using the `artwork_id` column.

```sparql
PREFIX emb: <https://artresearch.net/embeddings/>

SELECT ?uri ?title
WHERE {
  SERVICE <https://artresearch.net/embeddings/service> {
    ?work emb:data "https://example.com/image.jpg" .
    ?work emb:request_type "url" .
    ?work emb:model "clip,dino" .
    ?work emb:returnValues "artwork_id,uri" .
    ?work emb:joinOn "artwork_id" .
  }
}
LIMIT 20
```

**3. Text Search with Metadata Filtering**

This query searches for the text "a painting of a dog" and filters the results to only include items where the `repository` metadata attribute is "Rijksmuseum".

```sparql
PREFIX emb: <https://artresearch.net/embeddings/>

SELECT ?uri ?title
WHERE {
  SERVICE <https://artresearch.net/embeddings/service> {
    ?work emb:data "a painting of a dog" .
    ?work emb:request_type "text" .
    ?work emb:model "clip" .
    ?work emb:returnValues "uri,title" .
    ?work emb:filterBy "repository" .
    ?work emb:filterValues "Rijksmuseum" .
  }
}
LIMIT 10
```

## Configuration

The server is configured through command-line arguments. The following options are available:

-   `--model, -n`: The name of the model.
-   `--index, -i`: The path to the USearch index file.
-   `--metadata, -m`: The path to the metadata CSV file.
-   `--embedding_socket, -e`: The path to the embedding service socket.
-   `--port, -p`: The port for the search server. Defaults to `8545`.
-   `--threads, -j`: The number of threads to use. Defaults to the number of hardware cores.
-   `--sensitivity-defaults`: The default limits for sensitivity levels.
-   `--sensitivity-factors`: The standard deviation factors for sensitivity levels.
-   `--cache-size`: The size of the embedding cache.
-   `--model-uri-map`: A pair of model name and a path to a CSV file mapping URIs to local file paths.

You can specify multiple models by providing the `--model`, `--index`, `--metadata`, and `--embedding_socket` arguments multiple times. The number of arguments for each of these options must be the same.
