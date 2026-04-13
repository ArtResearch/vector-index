#ifndef IGNIS_SEARCH_SERVICE_HPP
#define IGNIS_SEARCH_SERVICE_HPP

#include "sparql/parser.hpp"
#include "model.hpp"
#include <map>

namespace ignis {
namespace search {

/**
 * @brief Orchestrates the entire search process.
 *
 * This class brings together all the components of the search pipeline. It takes a
 * parsed SPARQL query, manages the search across multiple models, invokes SIFT
 * reranking, and aggregates the results into a final, sorted list.
 */
class Service {
public:
    /**
     * @brief Constructs a search service.
     * @param models A map of model names to their corresponding ModelData.
     * @param sensitivity_defaults Default result limits for sensitivity levels.
     * @param sensitivity_factors Standard deviation factors for sensitivity levels.
     */
    Service(
        std::map<std::string, ModelData>& models,
        std::vector<int>& sensitivity_defaults,
        std::vector<float>& sensitivity_factors,
        const std::string& image_file_base_dir
    );

    /**
     * @brief Executes a search based on a parsed SPARQL query.
     * @param query The parsed SPARQL query.
     * @return A pair containing the sorted final results and a list of grouping models.
     */
    std::pair<
        std::vector<std::pair<std::string, FinalCombinedResult>>,
        std::vector<std::string>
    > search(const sparql::SparqlQuery& query);

private:
    std::map<std::string, ModelData>& _models;
    std::vector<int>& _sensitivity_defaults;
    std::vector<float>& _sensitivity_factors;
    std::string _image_file_base_dir;
};

} // namespace search
} // namespace ignis

#endif // IGNIS_SEARCH_SERVICE_HPP
