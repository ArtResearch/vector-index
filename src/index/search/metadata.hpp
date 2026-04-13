#ifndef IGNIS_SEARCH_METADATA_HPP
#define IGNIS_SEARCH_METADATA_HPP

#include <string>
#include "model.hpp"

namespace ignis {
namespace search {

/**
 * @brief Loads metadata from a CSV file into a ModelData struct.
 *
 * This function parses a CSV file, populates the metadata table, and builds
 * the necessary maps for efficient lookup (header map, URI to key map).
 *
 * @param metadata_file The path to the metadata CSV file.
 * @param model_data The ModelData object to populate.
 * @param indexed_metadata_columns A list of column names to build inverted indexes for.
 * @throws std::runtime_error if the file cannot be opened or if the 'uri' column is missing.
 */
void load_metadata(
    const std::string& metadata_file,
    ModelData& model_data,
    const std::vector<std::string>& indexed_metadata_columns
);

} // namespace search
} // namespace ignis

#endif // IGNIS_SEARCH_METADATA_HPP
