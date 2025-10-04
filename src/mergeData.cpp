#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <numeric>

#include <boost/program_options.hpp>
#include "npy.hpp"
#include "lazycsv.hpp"

namespace po = boost::program_options;

void merge_csv_files(const std::vector<std::string>& input_paths, const std::string& output_path) {
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        throw std::runtime_error("Failed to open output CSV file: " + output_path);
    }

    bool header_written = false;
    for (const auto& input_path : input_paths) {
        lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> parser(input_path);
        if (!header_written) {
            output_file << parser.header().raw() << "\n";
            header_written = true;
        }
        for (const auto& row : parser) {
            output_file << row.raw() << "\n";
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
