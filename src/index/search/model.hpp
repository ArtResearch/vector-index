#ifndef IGNIS_SEARCH_MODEL_HPP
#define IGNIS_SEARCH_MODEL_HPP

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>

#include "common/fifo_cache.hpp"
#include "common/types.hpp"

namespace ignis {
namespace search {

/// @brief A vector of strings representing a single row in a metadata CSV file.
using MetadataRow = std::vector<std::string>;

/**
 * @brief Holds the raw result from a SIFT matching operation before metadata lookups.
 */
struct SiftMatchResult {
    usearch_key_t key;
    float score;
};

/**
 * @brief Represents an intermediate result from a single model's search.
 *
 * This struct is used to aggregate results before they are combined into the final response.
 */
struct IntermediateResult {
    std::string group_key;
    std::string join_key;
    float score;
    std::string original_uri;
    std::string model_name;
};

/**
 * @brief Represents the final combined result for a given join key.
 *
 * It includes the final weighted score and a map of contributions from each model.
 */
struct FinalCombinedResult {
    float final_weighted_score = 0.0f;
    std::map<std::string, IntermediateResult> model_contributions;
};

/**
 * @brief Contains all data associated with a single search model.
 *
 * This includes the USearch index, metadata, caches, and configuration.
 */
struct ModelData {
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

} // namespace search
} // namespace ignis

#endif // IGNIS_SEARCH_MODEL_HPP
