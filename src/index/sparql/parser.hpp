#ifndef IGNIS_SPARQL_PARSER_HPP
#define IGNIS_SPARQL_PARSER_HPP

#include <string>
#include <vector>
#include <optional>

namespace ignis {
namespace sparql {

/**
 * @brief Represents a parsed SPARQL query from the client.
 *
 * This struct holds the extracted parameters from the raw query string in a
 * structured format, making them easy to use in the search service.
 */
struct SparqlQuery {
    std::string data;
    std::string request_type;
    size_t limit;
    std::vector<std::string> model_names;
    std::vector<std::string> join_on_columns;
    std::vector<std::string> return_values_columns;
    std::optional<std::string> sensitivity;
    bool exact_search;
    std::optional<std::string> filter_by;
    std::optional<std::string> filter_values;
};

/**
 * @brief Parses a raw SPARQL query string into a structured SparqlQuery object.
 *
 * This function uses regex to extract all the custom embedding parameters from the
 * SPARQL query.
 *
 * @param query The raw SPARQL query string.
 * @return An optional containing the parsed SparqlQuery. If parsing fails
 *         (e.g., a required parameter is missing), the optional will be empty.
 */
std::optional<SparqlQuery> parse_sparql_request(const std::string& query);

} // namespace sparql
} // namespace ignis

#endif // IGNIS_SPARQL_PARSER_HPP
