#ifndef IGNIS_TYPES_HPP
#define IGNIS_TYPES_HPP

#include <cstdint>
#include "usearch/index_dense.hpp"

/**
 * @file types.hpp
 * @brief Defines common type aliases used throughout the application.
 */

namespace ignis {

/// @brief The key type used in the USearch index.
using usearch_key_t = int64_t;

/// @brief The distance metric type (e.g., float for L2 distance).
using distance_t = float;

/// @brief The specific dense index type used for vector search.
using index_t = unum::usearch::index_dense_gt<usearch_key_t>;

/// @brief The result type returned from a dense index search.
using dense_search_result_t = typename index_t::search_result_t;

} // namespace ignis

#endif // IGNIS_TYPES_HPP
