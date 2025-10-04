#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>

#include <boost/program_options.hpp>
#include "usearch/index_dense.hpp"
#include "usearch/index_plugins.hpp"
#include "lazycsv.hpp"
#include "npy.hpp"

namespace po = boost::program_options;
using namespace unum::usearch;

// A struct to hold the metadata for a single row
using Metadata = std::map<std::string, std::string>;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("index-file", po::value<std::string>()->required(), "path to the USearch index file")
        ("metadata-file", po::value<std::string>()->required(), "path to the metadata csv file")
        ("embeddings-npy", po::value<std::string>()->required(), "path to embeddings numpy file")
        ("query-id", po::value<size_t>()->required(), "the ID of the vector to query for")
        ("count,k", po::value<size_t>()->default_value(10), "number of nearest neighbors to find");

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
        std::string index_path = vm["index-file"].as<std::string>();
        std::string metadata_path = vm["metadata-file"].as<std::string>();
        std::string embeddings_path = vm["embeddings-npy"].as<std::string>();
        size_t query_id = vm["query-id"].as<size_t>();
        size_t count = vm["count"].as<size_t>();

        // Load the USearch index
        index_dense_t index = index_dense_t::make(index_path.c_str(), true);
        if (index.size() == 0) {
            throw std::runtime_error("Index is empty or failed to load.");
        }
        std::cout << "Loaded index with " << index.size() << " vectors." << std::endl;

        // Load the metadata
        lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> metadata_parser(metadata_path);
        std::vector<std::string> header;
        for (const auto& cell : metadata_parser.header()) {
            header.push_back(std::string(cell.trimmed()));
        }

        std::vector<Metadata> metadata_table;
        for (const auto& row : metadata_parser) {
            Metadata metadata_row;
            int i = 0;
            for (const auto& cell : row) {
                metadata_row[header[i++]] = cell.unescaped();
            }
            metadata_table.push_back(metadata_row);
        }
        std::cout << "Loaded metadata for " << metadata_table.size() << " vectors." << std::endl;

        if (index.size() != metadata_table.size()) {
            std::cerr << "Warning: Index size and metadata size do not match." << std::endl;
        }

        // Load the query vector
        npy::npy_data<float> npy_data_float = npy::read_npy<float>(embeddings_path);
        if (query_id >= npy_data_float.shape[0]) {
            throw std::runtime_error("Query ID is out of bounds.");
        }
        size_t dimensions = npy_data_float.shape[1];
        std::vector<bf16_t> query_vector_bf16;
        query_vector_bf16.reserve(dimensions);
        for (size_t i = 0; i < dimensions; ++i) {
            query_vector_bf16.push_back(bf16_t(npy_data_float.data[query_id * dimensions + i]));
        }

        // Perform the search
        std::vector<uint64_t> found_keys(count);
        std::vector<float> found_distances(count);
        auto result = index.search(query_vector_bf16.data(), count);
        size_t found_count = result.dump_to(found_keys.data(), found_distances.data(), count);

        std::cout << "\nFound " << found_count << " nearest neighbors for vector " << query_id << ":" << std::endl;
        for (size_t i = 0; i < found_count; ++i) {
            size_t key = found_keys[i];
            float distance = found_distances[i];
            std::cout << "\nResult " << i + 1 << ":" << std::endl;
            std::cout << "  ID: " << key << ", Distance: " << distance << std::endl;
            if (key < metadata_table.size()) {
                const auto& metadata = metadata_table[key];
                for (const auto& pair : metadata) {
                    std::cout << "  " << pair.first << ": " << pair.second << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
