#include "parser.hpp"
#include <regex>
#include <boost/algorithm/string.hpp>

namespace ignis {
namespace sparql {

std::optional<SparqlQuery> parse_sparql_request(const std::string& query) {
    static const std::regex data_regex("(?:<https://artresearch\\.net/embeddings/data>|emb:data)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex request_type_regex("(?:<https://artresearch\\.net/embeddings/request_type>|emb:request_type)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex limit_regex("limit\\s+(\\d+)", std::regex_constants::icase);
    static const std::regex model_regex("(?:<https://artresearch\\.net/embeddings/model>|emb:model)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex joinon_regex("(?:<https://artresearch\\.net/embeddings/joinOn>|emb:joinOn)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex returnvalues_regex("(?:<https://artresearch\\.net/embeddings/returnValues>|emb:returnValues)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex sensitivity_regex("(?:<https://artresearch\\.net/embeddings/sensitivity>|emb:sensitivity)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex exact_regex("(?:<https://artresearch\\.net/embeddings/exact>|emb:exact)\\s*\"(true|false)\"", std::regex_constants::icase);
    static const std::regex filterby_regex("(?:<https://artresearch\\.net/embeddings/filterBy>|emb:filterBy)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex filtervalues_regex("(?:<https://artresearch\\.net/embeddings/filterValues>|emb:filterValues)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    // Sanitize filename to prevent directory traversal. Allows alphanumeric, underscore, dot, and dash.
    static const std::regex filename_regex("^[a-zA-Z0-9_.-]+$");

    std::smatch matches;
    SparqlQuery result;

    if (std::regex_search(query, matches, data_regex)) {
        result.data = matches[1].str();
    } else {
        return std::nullopt; // Data is required
    }

    if (std::regex_search(query, matches, request_type_regex)) {
        result.request_type = matches[1].str();
    } else {
        return std::nullopt; // Request type is required
    }

    if (result.request_type == "file") {
        if (!std::regex_match(result.data, filename_regex)) {
            return std::nullopt; // Invalid filename
        }
    }

    if (std::regex_search(query, matches, model_regex)) {
        std::string model_names_str = matches[1].str();
        boost::split(result.model_names, model_names_str, boost::is_any_of(","));
        for (auto& model_name : result.model_names) {
            boost::trim(model_name);
        }
    } else {
        return std::nullopt; // Model is required
    }
    if (result.model_names.empty()) {
        return std::nullopt;
    }

    result.limit = 10;
    if (std::regex_search(query, matches, limit_regex)) {
        result.limit = std::stoi(matches[1].str());
    }

    if (std::regex_search(query, matches, sensitivity_regex)) {
        result.sensitivity = matches[1].str();
    }

    if (std::regex_search(query, matches, filterby_regex)) {
        result.filter_by = matches[1].str();
    }

    if (std::regex_search(query, matches, filtervalues_regex)) {
        result.filter_values = matches[1].str();
    }

    result.exact_search = false;
    if (std::regex_search(query, matches, exact_regex)) {
        std::string exact_val = matches[1].str();
        boost::algorithm::to_lower(exact_val);
        if (exact_val == "true") {
            result.exact_search = true;
        }
    }

    std::string return_values_str = "uri";
    if (std::regex_search(query, matches, returnvalues_regex)) {
        return_values_str = matches[1].str();
    }
    boost::split(result.return_values_columns, return_values_str, boost::is_any_of(","));
    for (auto& col : result.return_values_columns) {
        boost::trim(col);
    }

    std::string join_on_str;
    if (std::regex_search(query, matches, joinon_regex)) {
        join_on_str = matches[1].str();
    } else {
        join_on_str = return_values_str;
    }
    boost::split(result.join_on_columns, join_on_str, boost::is_any_of(","));
    for (auto& col : result.join_on_columns) {
        boost::trim(col);
    }

    return result;
}

} // namespace sparql
} // namespace ignis
