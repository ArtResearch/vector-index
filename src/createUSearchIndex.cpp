#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <omp.h>

#include <boost/program_options.hpp>
#include "npy.hpp"
#include "usearch/index_dense.hpp"
#include "usearch/index_plugins.hpp"

namespace po = boost::program_options;
using namespace unum::usearch;

using usearch_key_t = int64_t;

template <typename scalar_t>
void create_index(const std::string& embeddings_path, const std::string& index_path, int threads) {
    if (threads == 0) {
        threads = omp_get_max_threads();
    }
    omp_set_num_threads(threads);
    std::cout << "Using " << threads << " threads for indexing." << std::endl;

    std::cout << "Loading embeddings from the .npy file." << std::endl;

    // Load embeddings from the .npy file
    npy::npy_data<float> npy_data_float = npy::read_npy<float>(embeddings_path);
    
    if (npy_data_float.shape.size() != 2) {
        throw std::runtime_error("Embeddings numpy array must be 2-dimensional.");
    }

    size_t num_vectors = npy_data_float.shape[0];
    size_t dimensions = npy_data_float.shape[1];
    
    std::vector<scalar_t> vectors_data_specialized;
    vectors_data_specialized.reserve(num_vectors * dimensions);
    for (const auto& val : npy_data_float.data) {
        vectors_data_specialized.push_back(scalar_t(val));
    }
    scalar_t* vectors_data = vectors_data_specialized.data();

    std::cout << "Loaded " << num_vectors << " embeddings with " << dimensions << " dimensions." << std::endl;

    // Initialize USearch index
    metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind<scalar_t>());
    index_dense_gt<usearch_key_t> index = index_dense_gt<usearch_key_t>::make(metric);
    
    index.reserve(num_vectors);

    // Add vectors to the index in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < num_vectors; ++i) {
        index.add(static_cast<usearch_key_t>(i), vectors_data + i * dimensions);
    }

    std::cout << "Index built with " << index.size() << " vectors." << std::endl;

    // Save the index to a file
    index.save(index_path.c_str());

    std::cout << "Index saved to " << index_path << std::endl;
}

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("embeddings-npy", po::value<std::string>()->required(), "path to embeddings numpy file")
        ("index-file", po::value<std::string>()->required(), "path to save the final index")
        ("threads,t", po::value<int>()->default_value(0), "number of threads to use for indexing (default: all available)")
        ("dtype", po::value<std::string>()->default_value("bf16"), "data type for the index (bf16 or f32)");

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
        std::string embeddings_path = vm["embeddings-npy"].as<std::string>();
        std::string index_path = vm["index-file"].as<std::string>();
        int threads = vm["threads"].as<int>();
        std::string dtype = vm["dtype"].as<std::string>();

        if (dtype == "bf16") {
            create_index<bf16_t>(embeddings_path, index_path, threads);
        } else if (dtype == "f32") {
            create_index<float>(embeddings_path, index_path, threads);
        } else {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
