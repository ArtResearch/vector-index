#include <iostream>
#include <thread>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>

#include "common/fifo_cache.hpp"
#include "lazycsv.hpp"
#include "search/model.hpp"
#include "search/metadata.hpp"
#include "search/service.hpp"
#include "http/server.hpp"
#include "http/handlers.hpp"

namespace po = boost::program_options;
namespace asio = boost::asio;

int main(int argc, char** argv) {
    po::options_description desc("C++ Search Server");
    desc.add_options()
        ("help,h", "produce help message")
        ("model,n", po::value<std::vector<std::string>>()->multitoken(), "model name")
        ("index,i", po::value<std::vector<std::string>>()->multitoken(), "path to the USearch index file")
        ("metadata,m", po::value<std::vector<std::string>>()->multitoken(), "path to the metadata CSV file")
        ("embedding_socket,e", po::value<std::vector<std::string>>()->multitoken(), "path to the embedding service socket")
        ("port,p", po::value<int>()->default_value(8545), "port for the search server")
        ("threads,j", po::value<int>()->default_value(0), "number of threads to use (0 = hardware concurrency)")
        ("sensitivity-defaults", po::value<std::string>()->default_value("100,1000,5000"), "default limits for sensitivity levels (precise,balanced,exploratory)")
        ("sensitivity-factors", po::value<std::string>()->default_value("0.25,-1.0,-2.0"), "std dev factors for sensitivity (precise,balanced,exploratory)")
        ("cache-size", po::value<int>()->default_value(1000), "size of the embedding cache per model")
        ("model-uri-map", po::value<std::vector<std::string>>()->multitoken(), "Pair of model name and path to the CSV mapping URIs to local file paths.")
        ("indexed-metadata-columns", po::value<std::string>()->default_value("uri"), "Comma-separated list of metadata columns to index for fast lookups.")
        ("uri-expansion-cap", po::value<int>()->default_value(1000), "Max number of keys to expand from a URI search.")
        ("image-file-base-dir", po::value<std::string>(), "Base directory for image files when using 'file' request type.");

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

    auto model_names = vm["model"].as<std::vector<std::string>>();
    auto index_files = vm["index"].as<std::vector<std::string>>();
    auto metadata_files = vm["metadata"].as<std::vector<std::string>>();
    auto embedding_sockets = vm["embedding_socket"].as<std::vector<std::string>>();
    
    std::vector<std::string> indexed_metadata_columns;
    std::string indexed_cols_str = vm["indexed-metadata-columns"].as<std::string>();
    boost::split(indexed_metadata_columns, indexed_cols_str, boost::is_any_of(","));
    for (auto& col : indexed_metadata_columns) {
        boost::trim(col);
    }

    std::map<std::string, std::string> model_uri_maps;
    if (vm.count("model-uri-map")) {
        const auto model_uri_map_pairs = vm["model-uri-map"].as<std::vector<std::string>>();
        if (model_uri_map_pairs.size() % 2 != 0) {
            std::cerr << "Error: --model-uri-map requires pairs of model name and file path." << std::endl;
            return 1;
        }
        for (size_t i = 0; i < model_uri_map_pairs.size(); i += 2) {
            model_uri_maps[model_uri_map_pairs[i]] = model_uri_map_pairs[i + 1];
        }
    }

    std::vector<int> sensitivity_defaults;
    std::string defaults_str = vm["sensitivity-defaults"].as<std::string>();
    std::vector<std::string> defaults_parts;
    boost::split(defaults_parts, defaults_str, boost::is_any_of(","));
    if (defaults_parts.size() == 3) {
        try {
            sensitivity_defaults.push_back(std::stoi(defaults_parts[0]));
            sensitivity_defaults.push_back(std::stoi(defaults_parts[1]));
            sensitivity_defaults.push_back(std::stoi(defaults_parts[2]));
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid format for sensitivity-defaults. Please use three comma-separated integers." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: sensitivity-defaults must contain three comma-separated integer values." << std::endl;
        return 1;
    }

    std::vector<float> sensitivity_factors;
    std::string factors_str = vm["sensitivity-factors"].as<std::string>();
    std::vector<std::string> factors_parts;
    boost::split(factors_parts, factors_str, boost::is_any_of(","));
    if (factors_parts.size() == 3) {
        try {
            sensitivity_factors.push_back(std::stof(factors_parts[0]));
            sensitivity_factors.push_back(std::stof(factors_parts[1]));
            sensitivity_factors.push_back(std::stof(factors_parts[2]));
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid format for sensitivity-factors. Please use three comma-separated floats." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: sensitivity-factors must contain three comma-separated float values." << std::endl;
        return 1;
    }

    if (model_names.size() != index_files.size() || model_names.size() != metadata_files.size() || model_names.size() != embedding_sockets.size()) {
        std::cerr << "Error: The number of models, indices, metadata files, and embedding sockets must be the same." << std::endl;
        return 1;
    }

    int cache_size = vm["cache-size"].as<int>();
    std::map<std::string, ignis::search::ModelData> models;

    for (size_t i = 0; i < model_names.size(); ++i) {
        try {
            std::cout << "Loading model " << model_names[i] << "..." << std::endl;
            ignis::search::ModelData model_data;
            model_data.embedding_socket_path = embedding_sockets[i];
            if (cache_size > 0) {
                model_data.embedding_cache = std::make_unique<fifo_cache<std::string, std::vector<float>>>(cache_size);
            }
            model_data.sift_cache = std::make_unique<fifo_cache<std::string, std::vector<ignis::search::SiftMatchResult>>>(1000);
            std::cout << "Loading index from " << index_files[i] << "..." << std::endl;
            model_data.index.load(index_files[i].c_str());
            std::cout << "Loading metadata from " << metadata_files[i] << "..." << std::endl;
            ignis::search::load_metadata(metadata_files[i], model_data, indexed_metadata_columns);
            
            auto it = model_uri_maps.find(model_names[i]);
            if (it != model_uri_maps.end()) {
                std::cout << "Loading URI to file map from " << it->second << " for model " << model_names[i] << "..." << std::endl;
                lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> uri_map_parser(it->second);
                for (const auto& row : uri_map_parser) {
                    model_data.uri_to_file_map[row.cells(0)[0].unescaped()] = row.cells(1)[0].unescaped();
                }
            }

            models[model_names[i]] = std::move(model_data);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load data for model " << model_names[i] << ": " << e.what() << std::endl;
            return 1;
        }
    }

    auto const port = static_cast<unsigned short>(vm["port"].as<int>());
    int threads = vm["threads"].as<int>();
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }

    auto const address = asio::ip::make_address("0.0.0.0");
    asio::io_context ioc{threads};

    std::string image_file_base_dir;
    if (vm.count("image-file-base-dir")) {
        image_file_base_dir = vm["image-file-base-dir"].as<std::string>();
    }

    auto service = std::make_shared<ignis::search::Service>(models, sensitivity_defaults, sensitivity_factors, image_file_base_dir);
    auto handler = ignis::http::create_handler(service);
    std::make_shared<ignis::http::listener>(ioc, asio::ip::tcp::endpoint{address, port}, handler)->run();

    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads > 1 ? threads - 1 : 0);
    for (auto i = threads - 1; i > 0; --i)
        threads_vec.emplace_back([&ioc] { ioc.run(); });
    
    std::cout << "Server started on port " << port << " with " << threads << " threads" << std::endl;
    ioc.run();

    return 0;
}
