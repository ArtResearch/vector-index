#include "response_formatter.hpp"

namespace ignis {
namespace sparql {

// Function to escape a string for JSON
static std::string escape_json(const std::string& s) {
    std::string escaped;
    escaped.reserve(s.length());
    for (char c : s) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    escaped += buf;
                } else {
                    escaped += c;
                }
                break;
        }
    }
    return escaped;
}

std::string format_sparql_response(
    const std::vector<std::pair<std::string, search::FinalCombinedResult>>& sorted_results,
    const std::vector<std::string>& grouping_models
) {
    std::string head_vars = R"("uri","similarity","matchedModel")";
    for (const auto& model_name : grouping_models) {
        head_vars += R"(,")" + model_name + R"(_maxScoreUri")";
    }

    std::string json = R"({"head":{"vars":[)" + head_vars + R"(]},"results":{"bindings":[)";

    for (const auto& pair : sorted_results) {
        const std::string& join_key = pair.first;
        const search::FinalCombinedResult& result = pair.second;

        json += R"({)";
        json += R"("uri":{"type":"uri","value":")" + escape_json(join_key) + R"("},)";
        json += R"("similarity":{"type":"literal","datatype":"http://www.w3.org/2001/XMLSchema#decimal","value":")" + std::to_string(result.final_weighted_score) + R"("},)";

        std::string matched_models_str;
        for (const auto& contribution_pair : result.model_contributions) {
            if (!matched_models_str.empty()) {
                matched_models_str += ",";
            }
            matched_models_str += contribution_pair.first;
        }
        json += R"("matchedModel":{"type":"literal","value":")" + escape_json(matched_models_str) + R"("},)";

        for (const auto& model_name : grouping_models) {
            auto it = result.model_contributions.find(model_name);
            if (it != result.model_contributions.end()) {
                json += R"(")" + model_name + R"(_maxScoreUri":{"type":"uri","value":")" + escape_json(it->second.original_uri) + R"("},)";
            }
        }
        
        if (json.back() == ',') json.pop_back();
        json += R"(},)";
    }

    if (json.back() == ',') json.pop_back();
    json += "]}}";
    return json;
}

} // namespace sparql
} // namespace ignis
