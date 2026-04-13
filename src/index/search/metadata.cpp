#include "metadata.hpp"
#include "lazycsv.hpp"
#include <stdexcept>
#include <unordered_set>
#include <boost/algorithm/string.hpp>

namespace ignis {
namespace search {

void load_metadata(
    const std::string& metadata_file,
    ModelData& model_data,
    const std::vector<std::string>& indexed_metadata_columns
) {
    lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> parser(metadata_file);

    // Load header
    for (const auto& cell : parser.header()) {
        model_data.metadata_header.push_back(cell.unescaped());
    }
    for (size_t i = 0; i < model_data.metadata_header.size(); ++i) {
        model_data.metadata_header_map[model_data.metadata_header[i]] = i;
    }

    auto uri_it = model_data.metadata_header_map.find("uri");
    if (uri_it == model_data.metadata_header_map.end()) {
        throw std::runtime_error("Metadata file must contain a 'uri' column.");
    }
    size_t uri_column_index = uri_it->second;

    // Create a set of column indices to be indexed for fast lookups
    std::unordered_set<size_t> indexed_column_indices;
    for (const auto& col_name : indexed_metadata_columns) {
        auto it = model_data.metadata_header_map.find(col_name);
        if (it != model_data.metadata_header_map.end()) {
            indexed_column_indices.insert(it->second);
        }
    }

    usearch_key_t current_key = 0;
    size_t row_index = 0;
    for (const auto& row : parser) {
        MetadataRow meta_row;
        std::string uri_str;
        
        size_t current_col_idx = 0;
        for (const auto& cell : row) {
            std::string cell_content = cell.unescaped();
            if (current_col_idx == uri_column_index) {
                uri_str = cell_content;
            }

            // If this column is indexed, populate the metadata_value_to_keys map
            if (indexed_column_indices.count(current_col_idx)) {
                const std::string& column_name = model_data.metadata_header[current_col_idx];
                std::vector<std::string> values;
                boost::split(values, cell_content, boost::is_any_of(","));
                for (auto& val : values) {
                    boost::trim(val);
                    if (!val.empty()) {
                        model_data.metadata_value_to_keys[column_name][val].push_back(current_key);
                    }
                }
            }
            meta_row.push_back(cell_content);
            current_col_idx++;
        }

        model_data.metadata_table.push_back(meta_row);
        if (!uri_str.empty()) {
            model_data.uri_to_key[uri_str] = current_key;
        }
        model_data.key_to_row_index[current_key] = row_index;

        current_key++;
        row_index++;
    }
}

} // namespace search
} // namespace ignis
