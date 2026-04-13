#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <numeric>
#include <sstream>
#include <map>
#include <algorithm>

#include <boost/program_options.hpp>
#include "npy.hpp"
#include "lazycsv.hpp"

namespace po = boost::program_options;

void merge_csv_files(const std::vector<std::string>& input_paths, const std::string& output_path) {
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output CSV file: " + output_path);
    }

    if (input_paths.empty()) {
        return;
    }

    lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> first_parser(input_paths[0]);
    std::vector<std::string> canonical_header;
    std::stringstream header_ss;
    bool first_col = true;
    for (const auto& cell : first_parser.header()) {
        canonical_header.push_back(std::string(cell.trimmed()));
        if (!first_col) {
            header_ss << ",";
        }
        // NOTE: We are quoting every cell. This assumes that the original data
        // does not contain quotes within the cells themselves.
        header_ss << "\"" << cell.raw() << "\"";
        first_col = false;
    }
    output_file << header_ss.str() << "\n";

    for (const auto& input_path : input_paths) {
        lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> parser(input_path);

        std::vector<std::string> current_header;
        for (const auto& cell : parser.header()) {
            current_header.push_back(std::string(cell.trimmed()));
        }

        if (current_header.size() != canonical_header.size()) {
            throw std::runtime_error("CSV files have different number of columns: " + input_path);
        }

        std::vector<int> column_mapping(canonical_header.size());
        for (size_t j = 0; j < canonical_header.size(); ++j) {
            auto it = std::find(current_header.begin(), current_header.end(), canonical_header[j]);
            if (it == current_header.end()) {
                throw std::runtime_error("Column '" + canonical_header[j] + "' not found in file: " + input_path);
            }
            column_mapping[j] = std::distance(current_header.begin(), it);
        }

        for (const auto& row : parser) {
            std::vector<std::string_view> cells;
            for(const auto& cell : row) {
                cells.push_back(cell.raw());
            }

            std::stringstream reordered_row_ss;
            for (size_t j = 0; j < column_mapping.size(); ++j) {
                // NOTE: We are quoting every cell. This assumes that the original data
                // does not contain quotes within the cells themselves.
                reordered_row_ss << "\"" << cells[column_mapping[j]] << "\"";
                if (j < column_mapping.size() - 1) {
                    reordered_row_ss << ",";
                }
            }
            output_file << reordered_row_ss.str() << "\n";
        }
    }
}

void merge_npy_files(const std::vector<std::string>& input_paths, const std::string& output_path) {
    std::vector<npy::npy_data<float>> npy_arrays;
    size_t total_rows = 0;
    size_t dimensions = 0;

    for (const auto& path : input_paths) {
        npy::npy_data<float> data = npy::read_npy<float>(path);
        if (data.shape.size() != 2) {
            throw std::runtime_error("NPY file is not 2-dimensional: " + path);
        }
        if (dimensions == 0) {
            dimensions = data.shape[1];
        } else if (dimensions != data.shape[1]) {
            throw std::runtime_error("NPY files have inconsistent dimensions.");
        }
        total_rows += data.shape[0];
        npy_arrays.push_back(std::move(data));
    }

    std::vector<float> merged_data;
    merged_data.reserve(total_rows * dimensions);

    for (const auto& data : npy_arrays) {
        merged_data.insert(merged_data.end(), data.data.begin(), data.data.end());
    }

    npy::npy_data<float> output_data;
    output_data.data = merged_data;
    output_data.shape = {total_rows, dimensions};
    npy::write_npy(output_path, output_data);
}

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("csv-files", po::value<std::vector<std::string>>()->multitoken()->required(), "input CSV files")
        ("npy-files", po::value<std::vector<std::string>>()->multitoken()->required(), "input NPY files")
        ("output-csv", po::value<std::string>()->required(), "output merged CSV file")
        ("output-npy", po::value<std::string>()->required(), "output merged NPY file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 1;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    try {
        auto csv_files = vm["csv-files"].as<std::vector<std::string>>();
        auto npy_files = vm["npy-files"].as<std::vector<std::string>>();
        auto output_csv = vm["output-csv"].as<std::string>();
        auto output_npy = vm["output-npy"].as<std::string>();

        if (csv_files.size() != npy_files.size()) {
            throw std::runtime_error("The number of CSV files must match the number of NPY files.");
        }

        for (size_t i = 0; i < csv_files.size(); ++i) {
            lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> parser(csv_files[i]);
            size_t csv_rows = std::distance(parser.begin(), parser.end());
            
            npy::npy_data<float> npy_data = npy::read_npy<float>(npy_files[i]);
            
            if (csv_rows != npy_data.shape[0]) {
                throw std::runtime_error("Mismatch between CSV rows and NPY vectors in file pair: " + csv_files[i] + " and " + npy_files[i]);
            }
        }

        std::cout << "Merging " << csv_files.size() << " CSV files..." << std::endl;
        merge_csv_files(csv_files, output_csv);
        std::cout << "Merged CSV file saved to " << output_csv << std::endl;

        std::cout << "Merging " << npy_files.size() << " NPY files..." << std::endl;
        merge_npy_files(npy_files, output_npy);
        std::cout << "Merged NPY file saved to " << output_npy << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
