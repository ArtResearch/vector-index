#include "service.hpp"
#include "embedding/client.hpp"
#include "image/sift_reranker.hpp"
#include "image/downloader.hpp"
#include "usearch/index_plugins.hpp"
#include <boost/beast/core/detail/base64.hpp>
#include <boost/algorithm/string.hpp>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

// Define MEASURE_PERFORMANCE to enable timing code
// You can control this with a compiler flag like -DMEASURE_PERFORMANCE
#ifdef MEASURE_PERFORMANCE
    #define START_TIMER(name) auto start##name = std::chrono::high_resolution_clock::now()
    #define END_TIMER(name) auto end##name = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<double, std::micro> elapsed##name = end##name - start##name; \
        std::cout << #name << " elapsed time: " << elapsed##name.count() << " us" << std::endl
#else
    #define START_TIMER(name)
    #define END_TIMER(name)
#endif

namespace ignis {
namespace search {

using unum::usearch::bf16_t;

constexpr size_t kSiftRerankCandidates = 100;

Service::Service(
    std::map<std::string, ModelData>& models,
    std::vector<int>& sensitivity_defaults,
    std::vector<float>& sensitivity_factors,
    const std::string& image_file_base_dir
) : _models(models), _sensitivity_defaults(sensitivity_defaults), _sensitivity_factors(sensitivity_factors), _image_file_base_dir(image_file_base_dir) {}

std::pair<
    std::vector<std::pair<std::string, FinalCombinedResult>>,
    std::vector<std::string>
> Service::search(const sparql::SparqlQuery& query) {
    START_TIMER(result_construction);
    std::vector<IntermediateResult> all_intermediate_results;
    std::vector<std::string> grouping_models_vec;

    if (query.request_type == "uri") {
        // URI-based search implementation
        for (size_t model_idx = 0; model_idx < query.model_names.size(); ++model_idx) {
            const auto& model_name = query.model_names[model_idx];
            auto it = _models.find(model_name);
            if (it == _models.end()) continue;
            ModelData& model_data = it->second;

            std::string join_column_name = (model_idx < query.join_on_columns.size()) ? query.join_on_columns[model_idx] : query.join_on_columns.back();
            std::string return_column_name = (model_idx < query.return_values_columns.size()) ? query.return_values_columns[model_idx] : query.return_values_columns.back();

            auto it_join_col = model_data.metadata_header_map.find(join_column_name);
            if (it_join_col == model_data.metadata_header_map.end()) {
                std::cerr << "Warning: joinOn column '" << join_column_name << "' not found in model '" << model_name << "'" << std::endl;
                continue;
            }
            size_t join_column_index = it_join_col->second;

            auto it_return_col = model_data.metadata_header_map.find(return_column_name);
            if (it_return_col == model_data.metadata_header_map.end()) {
                std::cerr << "Warning: returnValues column '" << return_column_name << "' not found in model '" << model_name << "'" << std::endl;
                continue;
            }
            size_t return_column_index = it_return_col->second;
            size_t uri_column_index = model_data.metadata_header_map.at("uri");

            // 1. Expand input identifier to a set of USearch keys
            std::vector<usearch_key_t> keys_to_search;
            if (join_column_name == "uri") {
                auto it_key = model_data.uri_to_key.find(query.data);
                if (it_key != model_data.uri_to_key.end()) {
                    keys_to_search.push_back(it_key->second);
                }
            } else {
                auto it_col = model_data.metadata_value_to_keys.find(join_column_name);
                if (it_col != model_data.metadata_value_to_keys.end()) {
                    auto it_val = it_col->second.find(query.data);
                    if (it_val != it_col->second.end()) {
                        keys_to_search = it_val->second;
                    }
                }
            }

            if (keys_to_search.empty()) continue;

            // 2. Compute seed exclusion sets
            std::unordered_set<usearch_key_t> seed_query_keys(keys_to_search.begin(), keys_to_search.end());
            std::unordered_set<std::string> seed_uri_set;
            std::unordered_set<std::string> seed_group_set_return;

            for (usearch_key_t seed_key : keys_to_search) {
                auto it_row = model_data.key_to_row_index.find(seed_key);
                if (it_row != model_data.key_to_row_index.end()) {
                    const auto& meta = model_data.metadata_table[it_row->second];
                    seed_uri_set.insert(meta[uri_column_index]);
                    if (return_column_name != "uri") {
                        seed_group_set_return.insert(meta[return_column_index]);
                    }
                }
            }

            // 3. Perform parallel search for each key
            std::vector<IntermediateResult> partial_results;
            partial_results.reserve(keys_to_search.size() * 100); // Pre-allocate
            size_t search_limit = 1000;

            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic) if (keys_to_search.size() > 1)
            #endif
            for (size_t i = 0; i < keys_to_search.size(); ++i) {
                usearch_key_t key = keys_to_search[i];
                std::vector<float> q(model_data.index.dimensions());
                if (model_data.index.get(key, q.data()) < 1) continue;

                auto knn = [&]() {
                    if (query.filter_by && query.filter_values && !query.filter_by->empty() && !query.filter_values->empty()) {
                        auto it_filter_col = model_data.metadata_header_map.find(*query.filter_by);
                        if (it_filter_col == model_data.metadata_header_map.end()) {
                            // If the filter column doesn't exist, return an empty result for this key
                            return index_t::search_result_t(model_data.index);
                        }
                        size_t filter_column_index = it_filter_col->second;

                        std::vector<std::string> filter_values_vec;
                        boost::split(filter_values_vec, *query.filter_values, boost::is_any_of(","));
                        std::unordered_set<std::string> filter_values_set(filter_values_vec.begin(), filter_values_vec.end());

                        auto predicate = [&](usearch_key_t key) -> bool {
                            auto it_row = model_data.key_to_row_index.find(key);
                            if (it_row == model_data.key_to_row_index.end()) return false;
                            const auto& meta = model_data.metadata_table[it_row->second];
                            const std::string& value_to_check = meta[filter_column_index];

                            std::string_view sv(value_to_check);
                            size_t start = 0;
                            while (start < sv.length()) {
                                size_t end = sv.find(',', start);
                                if (end == std::string_view::npos) {
                                    end = sv.length();
                                }
                                std::string_view token = sv.substr(start, end - start);

                                // Trim whitespace from token
                                size_t first = token.find_first_not_of(" \t\n\r");
                                if (std::string::npos == first) {
                                    token = std::string_view();
                                } else {
                                    size_t last = token.find_last_not_of(" \t\n\r");
                                    token = token.substr(first, (last - first + 1));
                                }

                                if (!token.empty() && filter_values_set.count(std::string(token)) > 0) {
                                    return true;
                                }
                                start = end + 1;
                            }
                            return false;
                        };
                        return model_data.index.filtered_search(q.data(), search_limit, predicate, index_t::any_thread(), query.exact_search);
                    } else {
                        return model_data.index.search(q.data(), search_limit, index_t::any_thread(), query.exact_search);
                    }
                }();


                std::vector<usearch_key_t> found_keys(knn.size());
                std::vector<distance_t> found_distances(knn.size());
                knn.dump_to(found_keys.data(), found_distances.data());

                float max_sim = 0.f;
                for (auto d : found_distances) max_sim = std::max(max_sim, 1.0f - d);
                if (max_sim <= 0.f) continue;

                std::vector<IntermediateResult> local_buffer;
                local_buffer.reserve(knn.size());

                for (size_t j = 0; j < knn.size(); ++j) {
                    usearch_key_t found_key = found_keys[j];
                    if (seed_query_keys.count(found_key)) continue;

                    auto it_row = model_data.key_to_row_index.find(found_key);
                    if (it_row == model_data.key_to_row_index.end()) continue;
                    
                    const auto& meta = model_data.metadata_table[it_row->second];
                    const std::string& candidate_uri = meta[uri_column_index];
                    if (seed_uri_set.count(candidate_uri)) continue;

                    IntermediateResult res;
                    res.group_key = meta[return_column_index];
                    res.join_key = meta[join_column_index];
                    res.score = (1.0f - found_distances[j]) / max_sim;
                    res.original_uri = candidate_uri;
                    res.model_name = model_name;
                    local_buffer.push_back(std::move(res));
                }

                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    partial_results.insert(partial_results.end(), local_buffer.begin(), local_buffer.end());
                }
            }

            // 4. Aggregation-time filtering and result aggregation
            bool needs_aggregation = (return_column_name != "uri");
            if (needs_aggregation) {
                grouping_models_vec.push_back(model_name);
                std::map<std::pair<std::string, std::string>, IntermediateResult> aggregated_results;
                for (const auto& res : partial_results) {
                    if (seed_group_set_return.count(res.group_key)) continue;
                    auto key = std::make_pair(res.group_key, res.model_name);
                    auto it_agg = aggregated_results.find(key);
                    if (it_agg == aggregated_results.end() || res.score > it_agg->second.score) {
                        aggregated_results[key] = res;
                    }
                }
                for (const auto& pair : aggregated_results) {
                    all_intermediate_results.push_back(pair.second);
                }
            } else {
                for (const auto& res : partial_results) {
                    // For URI returns, the seed_uri_set check at match time is sufficient
                    all_intermediate_results.push_back(res);
                }
            }
        }
    } else { // Existing text/image search path
        bool is_image_request = (query.request_type == "url" || query.request_type == "file");

        for (size_t model_idx = 0; model_idx < query.model_names.size(); ++model_idx) {
            const auto& model_name = query.model_names[model_idx];
            std::string return_column_name = (model_idx < query.return_values_columns.size()) ? query.return_values_columns[model_idx] : query.return_values_columns.back();
            bool needs_aggregation = (return_column_name != "uri");
            if (needs_aggregation) {
                grouping_models_vec.push_back(model_name);
                if (is_image_request) {
                    grouping_models_vec.push_back(model_name + "_sift_rerank");
                }
            }
        }

        for (size_t model_idx = 0; model_idx < query.model_names.size(); ++model_idx) {
            const auto& model_name = query.model_names[model_idx];
            auto it = _models.find(model_name);
            if (it == _models.end()) continue;
            ModelData& model_data = it->second;

            size_t uri_column_index = model_data.metadata_header_map.at("uri");
            std::string join_column_name = (model_idx < query.join_on_columns.size()) ? query.join_on_columns[model_idx] : query.join_on_columns.back();
            std::string return_column_name = (model_idx < query.return_values_columns.size()) ? query.return_values_columns[model_idx] : query.return_values_columns.back();
            
            auto it_join_col = model_data.metadata_header_map.find(join_column_name);
            if (it_join_col == model_data.metadata_header_map.end()) continue;
            size_t join_column_index = it_join_col->second;

            auto it_return_col = model_data.metadata_header_map.find(return_column_name);
            if (it_return_col == model_data.metadata_header_map.end()) continue;
            size_t return_column_index = it_return_col->second;

            START_TIMER(embedding_generation);
            std::vector<float> embedding;
            cv::Mat image_mat; // To hold the decoded and resized image

            try {
                if (is_image_request) {
                    // 1. Get raw image data
                    std::string raw_image_data;
                    if (query.request_type == "url") {
                        raw_image_data = image::download_file(query.data);
                    } else { // "file"
                        if (_image_file_base_dir.empty()) {
                            throw std::runtime_error("Image file base directory is not configured.");
                        }
                        std::string file_path = _image_file_base_dir + "/" + query.data;
                        std::ifstream file(file_path, std::ios::binary);
                        if (!file) {
                            throw std::runtime_error("Failed to open image file: " + file_path);
                        }
                        file.seekg(0, std::ios::end);
                        raw_image_data.resize(file.tellg());
                        file.seekg(0, std::ios::beg);
                        file.read(&raw_image_data[0], raw_image_data.size());
                    }

                    // 2. Decode
                    std::vector<char> data_vec(raw_image_data.begin(), raw_image_data.end());
                    image_mat = cv::imdecode(data_vec, cv::IMREAD_COLOR);
                    if (image_mat.empty()) {
                        throw std::runtime_error("Failed to decode image in search service");
                    }

                    // 3. Resize if necessary
                    const int max_dimension = 512;
                    if (image_mat.cols > max_dimension || image_mat.rows > max_dimension) {
                        double scale = (image_mat.cols > image_mat.rows) ?
                            static_cast<double>(max_dimension) / image_mat.cols :
                            static_cast<double>(max_dimension) / image_mat.rows;
                        cv::resize(image_mat, image_mat, cv::Size(), scale, scale, cv::INTER_LANCZOS4);
                    }

                    // 4. Get embedding
                    // Always send the original raw image data, not a re-encoded or resized version.
                    // The python server is responsible for resizing.
                    embedding = embedding::get_embedding_from_socket(raw_image_data, embedding::PayloadType::Image, model_data);
                } else { // "text"
                    embedding = embedding::get_embedding_from_socket(query.data, model_data);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error getting embedding for model " << model_name << ": " << e.what() << std::endl;
                continue;
            }
            END_TIMER(embedding_generation);

            std::vector<IntermediateResult> current_model_results;
            size_t search_limit = (query.sensitivity.has_value() && query.sensitivity.value() == "near-exact") ? kSiftRerankCandidates : 1000;
            START_TIMER(actual_search);

            auto result = [&]() {
                auto perform_search = [&](auto embedding_ptr) {
                    if (query.filter_by && query.filter_values && !query.filter_by->empty() && !query.filter_values->empty()) {
                        auto it_filter_col = model_data.metadata_header_map.find(*query.filter_by);
                        if (it_filter_col == model_data.metadata_header_map.end()) {
                            return index_t::search_result_t(model_data.index);
                        }
                        size_t filter_column_index = it_filter_col->second;

                        std::vector<std::string> filter_values_vec;
                        boost::split(filter_values_vec, *query.filter_values, boost::is_any_of(","));
                        std::unordered_set<std::string> filter_values_set(filter_values_vec.begin(), filter_values_vec.end());

                        auto predicate = [&](usearch_key_t key) -> bool {
                            auto it_row = model_data.key_to_row_index.find(key);
                            if (it_row == model_data.key_to_row_index.end()) return false;
                            const auto& meta = model_data.metadata_table[it_row->second];
                            const std::string& value_to_check = meta[filter_column_index];

                            std::string_view sv(value_to_check);
                            size_t start = 0;
                            while (start < sv.length()) {
                                size_t end = sv.find(',', start);
                                if (end == std::string_view::npos) {
                                    end = sv.length();
                                }
                                std::string_view token = sv.substr(start, end - start);

                                // Trim whitespace from token
                                size_t first = token.find_first_not_of(" \t\n\r");
                                if (std::string::npos == first) {
                                    token = std::string_view();
                                } else {
                                    size_t last = token.find_last_not_of(" \t\n\r");
                                    token = token.substr(first, (last - first + 1));
                                }

                                if (!token.empty() && filter_values_set.count(std::string(token)) > 0) {
                                    return true;
                                }
                                start = end + 1;
                            }
                            return false;
                        };
                        return model_data.index.filtered_search(embedding_ptr, search_limit, predicate, index_t::any_thread(), query.exact_search);
                    } else {
                        return model_data.index.search(embedding_ptr, search_limit, index_t::any_thread(), query.exact_search);
                    }
                };

                if (model_data.index.scalar_kind() == unum::usearch::scalar_kind_t::bf16_k) {
                    // The USearch index is built with bfloat16 vectors, but the embedding
                    // service returns float32 vectors. We must convert the query vector to
                    // bfloat16 to match the index's data type for accurate distance calculation.
                    std::vector<bf16_t> embedding_bf16;
                    embedding_bf16.reserve(embedding.size());
                    for (float val : embedding) {
                        embedding_bf16.push_back(bf16_t(val));
                    }
                    return perform_search(embedding_bf16.data());
                } else if (model_data.index.scalar_kind() == unum::usearch::scalar_kind_t::f32_k) {
                    return perform_search(embedding.data());
                } else {
                    throw std::runtime_error("Unsupported index scalar kind");
                }
            }();
            END_TIMER(actual_search);

            if (query.sensitivity.has_value() && query.sensitivity.value() == "near-exact") {
                if (query.model_names.size() > 1) throw std::runtime_error("near-exact sensitivity can only be used with a single model");
                if (!is_image_request) throw std::runtime_error("near-exact sensitivity can only be used with image requests");
                
                if (result.size() > 0) {
                    std::vector<usearch_key_t> found_keys(result.size());
                    std::vector<distance_t> found_distances(result.size());
                    result.dump_to(found_keys.data(), found_distances.data());
                    
                    std::vector<SiftMatchResult> raw_sift_results;
                    std::string sift_model_name = model_name + "_sift_rerank";
                    std::size_t data_hash = std::hash<std::string>{}(query.data);
                    std::string cache_key = model_name + ":" + std::to_string(data_hash);

                    if (model_data.sift_cache && model_data.sift_cache->exists(cache_key)) {
                        raw_sift_results = model_data.sift_cache->get(cache_key);
                    } else {
                        START_TIMER(sift_reranking);
                        raw_sift_results = image::rerank_with_sift(image_mat, found_keys, found_distances, model_data, uri_column_index);
                        END_TIMER(sift_reranking);
                        if (model_data.sift_cache) {
                            model_data.sift_cache->put(cache_key, raw_sift_results);
                        }
                    }

                    if (!raw_sift_results.empty()) {
                        float max_sift_score = 0.0f;
                        for(const auto& res : raw_sift_results) max_sift_score = std::max(max_sift_score, res.score);

                        if (max_sift_score > 0.0f) {
                            for (const auto& raw_res : raw_sift_results) {
                                const auto& meta = model_data.metadata_table[raw_res.key];
                                IntermediateResult res;
                                res.group_key = meta[return_column_index];
                                res.join_key = meta[join_column_index];
                                res.score = raw_res.score / max_sift_score;
                                res.original_uri = meta[uri_column_index];
                                res.model_name = sift_model_name;
                                current_model_results.push_back(res);
                            }
                        }
                    }
                }
            } else {
                if (result.size() > 0) {
                    std::vector<usearch_key_t> found_keys(result.size());
                    std::vector<distance_t> found_distances(result.size());
                    result.dump_to(found_keys.data(), found_distances.data());

                    if (is_image_request) {
                        std::vector<usearch_key_t> keys_for_sift;
                        std::vector<distance_t> distances_for_sift;
                        keys_for_sift.reserve(kSiftRerankCandidates);
                        distances_for_sift.reserve(kSiftRerankCandidates);

                        for (size_t i = 0; i < std::min(kSiftRerankCandidates, result.size()); ++i) {
                            if (1.0f - found_distances[i] >= 0.75f) {
                                keys_for_sift.push_back(found_keys[i]);
                                distances_for_sift.push_back(found_distances[i]);
                            }
                        }

                        if (!keys_for_sift.empty()) {
                            std::vector<SiftMatchResult> raw_sift_results;
                            std::string sift_model_name = model_name + "_sift_rerank";
                            std::size_t data_hash = std::hash<std::string>{}(query.data);
                            std::string cache_key = model_name + ":" + std::to_string(data_hash);

                            if (model_data.sift_cache && model_data.sift_cache->exists(cache_key)) {
                                raw_sift_results = model_data.sift_cache->get(cache_key);
                            } else {
                                START_TIMER(sift_reranking);
                                raw_sift_results = image::rerank_with_sift(image_mat, keys_for_sift, distances_for_sift, model_data, uri_column_index);
                                END_TIMER(sift_reranking);
                                if (model_data.sift_cache) {
                                    model_data.sift_cache->put(cache_key, raw_sift_results);
                                }
                            }

                            if (!raw_sift_results.empty()) {
                                float max_sift_score = 0.0f;
                                for(const auto& res : raw_sift_results) max_sift_score = std::max(max_sift_score, res.score);

                                if (max_sift_score > 0.0f) {
                                    for (const auto& raw_res : raw_sift_results) {
                                        const auto& meta = model_data.metadata_table[raw_res.key];
                                        IntermediateResult res;
                                        res.group_key = meta[return_column_index];
                                        res.join_key = meta[join_column_index];
                                        res.score = raw_res.score / max_sift_score;
                                        res.original_uri = meta[uri_column_index];
                                        res.model_name = sift_model_name;
                                        current_model_results.push_back(res);
                                    }
                                }
                            }
                        }
                    }

                    float max_similarity = 0.0f;
                    for (distance_t dist : found_distances) max_similarity = std::max(max_similarity, 1.0f - dist);
                    
                    if (max_similarity > 0.0f) {
                        for (size_t i = 0; i < result.size(); ++i) {
                            const auto& meta = model_data.metadata_table[found_keys[i]];
                            IntermediateResult res;
                            res.group_key = meta[return_column_index];
                            res.join_key = meta[join_column_index];
                            res.score = (1.0f - found_distances[i]) / max_similarity;
                            res.original_uri = meta[uri_column_index];
                            res.model_name = model_name;
                            current_model_results.push_back(res);
                        }
                    }
                }
            }

            bool needs_aggregation = (return_column_name != "uri");
            if (needs_aggregation) {
                std::map<std::pair<std::string, std::string>, IntermediateResult> aggregated_results;
                for (const auto& res : current_model_results) {
                    auto key = std::make_pair(res.group_key, res.model_name);
                    auto it_agg = aggregated_results.find(key);
                    if (it_agg == aggregated_results.end() || res.score > it_agg->second.score) {
                        aggregated_results[key] = res;
                    }
                }
                for (const auto& pair : aggregated_results) {
                    all_intermediate_results.push_back(pair.second);
                }
            } else {
                all_intermediate_results.insert(all_intermediate_results.end(), current_model_results.begin(), current_model_results.end());
            }
        }
    }

    // Final result processing (common to both URI and text/image paths)
    std::map<std::string, FinalCombinedResult> final_results;
    bool has_sift_results = false;
    for (const auto& res : all_intermediate_results) {
        if (boost::ends_with(res.model_name, "_sift_rerank")) {
            has_sift_results = true;
            break;
        }
    }

    for (const auto& intermediate_res : all_intermediate_results) {
        std::vector<std::string> join_keys;
        boost::split(join_keys, intermediate_res.join_key, boost::is_any_of(","));

        for (const auto& key : join_keys) {
            if (key.empty()) continue;

            float weight = 1.0f;
            if (has_sift_results) {
                if (boost::ends_with(intermediate_res.model_name, "_sift_rerank")) {
                    weight = 0.9f;
                } else {
                    weight = 0.1f;
                }
            } else {
                weight = 1.0f / query.model_names.size();
            }

            FinalCombinedResult& final_res = final_results[key];
            final_res.final_weighted_score += intermediate_res.score * weight;
            
            std::string contribution_key = intermediate_res.model_name;
            auto it_contrib = final_res.model_contributions.find(contribution_key);
            if (it_contrib == final_res.model_contributions.end() || intermediate_res.score > it_contrib->second.score) {
                 final_res.model_contributions[contribution_key] = intermediate_res;
            }
        }
    }

    std::vector<std::pair<std::string, FinalCombinedResult>> sorted_results(final_results.begin(), final_results.end());
    std::sort(sorted_results.begin(), sorted_results.end(), [](const auto& a, const auto& b) {
        return a.second.final_weighted_score > b.second.final_weighted_score;
    });

    if (query.sensitivity.has_value() && query.sensitivity.value() != "near-exact" && sorted_results.size() > 10) {
        std::vector<float> non_sift_scores;
        non_sift_scores.reserve(sorted_results.size());
        for(const auto& pair : sorted_results) {
            bool has_sift_contribution = false;
            for (const auto& contrib_pair : pair.second.model_contributions) {
                if (boost::ends_with(contrib_pair.first, "_sift_rerank")) {
                    has_sift_contribution = true;
                    break;
                }
            }
            if (!has_sift_contribution) {
                non_sift_scores.push_back(pair.second.final_weighted_score);
            }
        }

        float cutoff_score = -1.0f;
        if (!non_sift_scores.empty()) {
            float sum = std::accumulate(non_sift_scores.begin(), non_sift_scores.end(), 0.0f);
            float mean = sum / non_sift_scores.size();
            float sq_sum = std::inner_product(non_sift_scores.begin(), non_sift_scores.end(), non_sift_scores.begin(), 0.0f);
            float std_dev = std::sqrt(sq_sum / non_sift_scores.size() - mean * mean);

            if (query.sensitivity.value() == "precise") {
                cutoff_score = mean + _sensitivity_factors[0] * std_dev;
            } else if (query.sensitivity.value() == "balanced") {
                cutoff_score = mean + _sensitivity_factors[1] * std_dev;
            } else if (query.sensitivity.value() == "exploratory") {
                cutoff_score = mean + _sensitivity_factors[2] * std_dev;
            }
        }

        std::vector<std::pair<std::string, FinalCombinedResult>> trimmed_results;
        for (const auto& pair : sorted_results) {
            bool has_sift_contribution = false;
            for (const auto& contrib_pair : pair.second.model_contributions) {
                if (boost::ends_with(contrib_pair.first, "_sift_rerank")) {
                    has_sift_contribution = true;
                    break;
                }
            }

            if (has_sift_contribution || pair.second.final_weighted_score >= cutoff_score) {
                trimmed_results.push_back(pair);
            }
        }
        
        size_t new_size = trimmed_results.size();
        if (query.sensitivity.value() == "precise" && new_size < static_cast<size_t>(_sensitivity_defaults[0])) new_size = std::min(static_cast<size_t>(_sensitivity_defaults[0]), sorted_results.size());
        else if (query.sensitivity.value() == "balanced" && new_size < static_cast<size_t>(_sensitivity_defaults[1])) new_size = std::min(static_cast<size_t>(_sensitivity_defaults[1]), sorted_results.size());
        else if (query.sensitivity.value() == "exploratory" && new_size < static_cast<size_t>(_sensitivity_defaults[2])) new_size = std::min(static_cast<size_t>(_sensitivity_defaults[2]), sorted_results.size());

        sorted_results = trimmed_results;
        if (sorted_results.size() > new_size) {
            sorted_results.resize(new_size);
        }

    } else if (query.sensitivity.has_value() && query.sensitivity.value() != "near-exact") {
        size_t default_limit = sorted_results.size();
        if (query.sensitivity.value() == "precise") default_limit = _sensitivity_defaults[0];
        else if (query.sensitivity.value() == "balanced") default_limit = _sensitivity_defaults[1];
        else if (query.sensitivity.value() == "exploratory") default_limit = _sensitivity_defaults[2];
        
        if (sorted_results.size() > default_limit) {
            sorted_results.resize(default_limit);
        }
    }

    END_TIMER(result_construction);
    return {sorted_results, grouping_models_vec};
}

} // namespace search
} // namespace ignis
