#ifndef IGNIS_SPARQL_RESPONSE_FORMATTER_HPP
#define IGNIS_SPARQL_RESPONSE_FORMATTER_HPP

#include <string>
#include <vector>
#include <utility>
#include "search/model.hpp"

namespace ignis {
namespace sparql {

/**
 * @brief Formats the final search results into a SPARQL 1.1 Query Results JSON Format.
 *
 * @param sorted_results A vector of pairs, where each pair contains a join key and
 *                       the corresponding final combined result. The vector must be
 *                       sorted by score in descending order.
 * @param grouping_models A list of model names that were used for grouping/aggregation.
 * @return A string containing the formatted JSON response.
 */
std::string format_sparql_response(
    const std::vector<std::pair<std::string, search::FinalCombinedResult>>& sorted_results,
    const std::vector<std::string>& grouping_models
);

} // namespace sparql
} // namespace ignis

#endif // IGNIS_SPARQL_RESPONSE_FORMATTER_HPP
