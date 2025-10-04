#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>
#include <charconv>
#include <thread>
#include <regex>
#include <unordered_map>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <boost/url.hpp>
#include <boost/beast/core/detail/base64.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <numeric>
#include <curl/curl.h>
#include <memory>
#include <fstream>
#include <optional>
#include <boost/circular_buffer.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <functional> // For std::hash
#include "usearch/index_dense.hpp"
#include "usearch/index_plugins.hpp"
#include "lazycsv.hpp"




template<typename key_t, typename value_t>
class fifo_cache {
public:
    fifo_cache(size_t max_size) :
        _keys(max_size) {
    }

    void put(const key_t& key, const value_t& value) {
        if (_cache_items_map.find(key) == _cache_items_map.end()) {
            if (_keys.full()) {
                _cache_items_map.erase(_keys.front());
            }
            _keys.push_back(key);
            _cache_items_map[key] = value;
        }
    }

    const value_t& get(const key_t& key) {
        auto it = _cache_items_map.find(key);
        if (it == _cache_items_map.end()) {
            throw std::range_error("There is no such key in cache");
        }
        return it->second;
    }

    bool exists(const key_t& key) const {
        return _cache_items_map.find(key) != _cache_items_map.end();
    }

    size_t size() const {
        return _cache_items_map.size();
    }

private:
    std::unordered_map<key_t, value_t> _cache_items_map;
    boost::circular_buffer<key_t> _keys;
};

// Define MEASURE_PERFORMANCE to enable timing code
// You can control this with a compiler flag like -DMEASURE_PERFORMANCE
#ifdef MEASURE_PERFORMANCE
    #define START_TIMER(name) auto start_##name = std::chrono::high_resolution_clock::now()
    #define END_TIMER(name) auto end_##name = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<double, std::micro> elapsed_##name = end_##name - start_##name; \
        std::cout << #name << " elapsed time: " << elapsed_##name.count() << " us" << std::endl
#else
    #define START_TIMER(name)
    #define END_TIMER(name)
#endif

namespace po = boost::program_options;
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace urls = boost::urls;
using tcp = boost::asio::ip::tcp;
using namespace unum::usearch;

using usearch_key_t = int64_t;
using distance_t = float;
using index_t = index_dense_gt<usearch_key_t>;
using dense_search_result_t = typename index_t::search_result_t;

// A struct to hold the metadata for a single row
using MetadataRow = std::vector<std::string>;

struct IntermediateResult {
    std::string group_key;
    std::string join_key;
    float score;
    std::string original_uri;
    std::string model_name;
};

// Holds the raw result from SIFT matching, before metadata lookups
struct SiftMatchResult {
    usearch_key_t key;
    float score;
};

struct ModelData {
    index_t index;
    std::vector<MetadataRow> metadata_table;
    std::vector<std::string> metadata_header;
    std::unordered_map<std::string, size_t> metadata_header_map;
    std::unordered_map<std::string, usearch_key_t> uri_to_key;
    std::string embedding_socket_path;
    std::unique_ptr<fifo_cache<std::string, std::vector<float>>> embedding_cache;
    std::unique_ptr<fifo_cache<std::string, std::vector<SiftMatchResult>>> sift_cache;
    std::unordered_map<std::string, std::string> uri_to_file_map;
};

struct FinalCombinedResult {
    float final_weighted_score = 0.0f;
    std::map<std::string, IntermediateResult> model_contributions;
};


// Global map to hold all models
std::map<std::string, ModelData> models;
std::vector<int> sensitivity_defaults;
std::vector<float> sensitivity_factors;


// Function to escape a string for JSON
std::string escape_json(const std::string& s) {
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
                    // Control characters must be escaped
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

// Function to get embedding from Python server via Unix socket
std::vector<float> get_embedding_from_socket(const std::string& payload, ModelData& model_data) {
    if (model_data.embedding_cache && model_data.embedding_cache->exists(payload)) {
        return model_data.embedding_cache->get(payload);
    }

    asio::io_context io_context;
    asio::local::stream_protocol::socket s(io_context);
    s.connect(model_data.embedding_socket_path);

    // Send the length of the payload first, in network byte order
    uint32_t payload_len_net = htonl(payload.length());
    asio::write(s, asio::buffer(&payload_len_net, sizeof(payload_len_net)));

    // Send payload to Python server
    asio::write(s, asio::buffer(payload.c_str(), payload.length()));

    // Read embedding size
    uint32_t embedding_size_net;
    asio::read(s, asio::buffer(&embedding_size_net, sizeof(embedding_size_net)));
    uint32_t embedding_size = ntohl(embedding_size_net);

    if (embedding_size == 0) {
        throw std::runtime_error("Failed to get embedding from server.");
    }

    // Read embedding data
    std::vector<float> embedding(embedding_size);
    asio::read(s, asio::buffer(embedding.data(), embedding_size * sizeof(float)));

    if (model_data.embedding_cache) {
        model_data.embedding_cache->put(payload, embedding);
    }

    return embedding;
}

// Function to load metadata from a CSV file
void load_metadata(const std::string& metadata_file, ModelData& model_data) {
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

    usearch_key_t current_key = 0;
    for (const auto& row : parser) {
        MetadataRow meta_row;
        std::string uri_str;
        size_t current_col_idx = 0;
        for (const auto& cell : row) {
            std::string cell_content = cell.unescaped();
            if (current_col_idx == uri_column_index) {
                uri_str = cell_content;
            }
            meta_row.push_back(cell_content);
            current_col_idx++;
        }
        model_data.metadata_table.push_back(meta_row);
        if (!uri_str.empty()) {
            model_data.uri_to_key[uri_str] = current_key;
        }
        current_key++;
    }
}


std::string format_sparql_response_new(
    const std::vector<std::pair<std::string, FinalCombinedResult>>& sorted_results, 
    const std::vector<std::string>& grouping_models
) {
    // Build variable list for the header
    std::string head_vars = R"("uri","similarity","matchedModel")";
    for (const auto& model_name : grouping_models) {
        head_vars += R"(,")" + model_name + R"(_maxScoreUri")";
    }

    std::string json = R"({"head":{"vars":[)" + head_vars + R"(]},"results":{"bindings":[)";

    for (const auto& pair : sorted_results) {
        const std::string& join_key = pair.first;
        const FinalCombinedResult& result = pair.second;

        json += R"({)";
        // URI is the join key
        json += R"("uri":{"type":"uri","value":")" + escape_json(join_key) + R"("},)";
        // Final weighted score
        json += R"("similarity":{"type":"literal","datatype":"http://www.w3.org/2001/XMLSchema#decimal","value":")" + std::to_string(result.final_weighted_score) + R"("},)";

        // Matched models
        std::string matched_models_str;
        for (const auto& contribution_pair : result.model_contributions) {
            if (!matched_models_str.empty()) {
                matched_models_str += ",";
            }
            matched_models_str += contribution_pair.first;
        }
        json += R"("matchedModel":{"type":"literal","value":")" + escape_json(matched_models_str) + R"("},)";

        // Dynamic bindings for each model's max score URI
        for (const auto& model_name : grouping_models) {
            auto it = result.model_contributions.find(model_name);
            if (it != result.model_contributions.end()) {
                json += R"(")" + model_name + R"(_maxScoreUri":{"type":"uri","value":")" + escape_json(it->second.original_uri) + R"("},)";
            }
        }
        
        if (json.back() == ',') json.pop_back(); // Remove last comma
        json += R"(},)";
    }

    if (json.back() == ',') json.pop_back();
    json += "]}}";
    // std::cout << "Final JSON: " << json << std::endl;
    return json;
}

// Function to convert a matrix of SIFT descriptors to RootSIFT
void compute_rootsift(cv::Mat& descriptors) {
    if (descriptors.empty()) {
        return;
    }

    // This function modifies the descriptors in-place.
    // Ensure descriptors are of type CV_32F (which SIFT produces).
    if (descriptors.type() != CV_32F) {
        descriptors.convertTo(descriptors, CV_32F);
    }

    // 1. L1-normalize each descriptor
    for (int i = 0; i < descriptors.rows; ++i) {
        double sum = cv::norm(descriptors.row(i), cv::NORM_L1);
        if (sum > 0) {
            descriptors.row(i) /= sum;
        }
    }

    // 2. Take the square root of each element
    cv::sqrt(descriptors, descriptors);

    // Note: The original paper mentions a final L2 normalization,
    // but many implementations, including the one you found and
    // the original authors' code, find it's often not necessary
    // for matching with BFMatcher as distance ranking remains the same.
    // We will omit it here for simplicity and speed, mirroring the Python code.
}

// DEBUG HELPER: Dumps a cv::Mat to a text file for inspection.
void dumpMatrix(const std::string& filename, const cv::Mat& mat) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    file << mat.rows << " " << mat.cols << " " << mat.type() << std::endl;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            file << mat.at<float>(i, j) << " ";
        }
        file << std::endl;
    }
    file.close();
}

// Callback function to write received data into a string
size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t new_length = size * nmemb;
    try {
        s->append((char*)contents, new_length);
    } catch (std::bad_alloc& e) {
        // handle memory problem
        return 0;
    }
    return new_length;
}

// Function to download a file from a URL
std::string download_file(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Could not initialize curl");
    }

    std::string readBuffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "artresearch.net/1.0 (https://artresearch.net; info@artresearch.net) Mozilla/5.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, ""); // Allow all supported compressions
    curl_easy_setopt(curl, CURLOPT_MAXFILESIZE_LARGE, (curl_off_t)(20 * 1024 * 1024)); // Set 20MB limit

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::string error_msg = "curl_easy_perform() failed: ";
        error_msg += curl_easy_strerror(res);
        curl_easy_cleanup(curl);
        throw std::runtime_error(error_msg);
    }

    curl_easy_cleanup(curl);
    return readBuffer;
}

// New function for SIFT reranking
std::vector<SiftMatchResult> rerank_with_sift(
    const std::string& data,
    const std::string& request_type,
    const std::vector<usearch_key_t>& candidate_keys,
    const std::vector<distance_t>& candidate_distances,
    ModelData& model_data,
    size_t uri_column_index
) {
    const int MIN_VOTE_THRESHOLD = 15; // As per proposal
    const int MIN_INLIERS_CUTOFF = 15; // Keep a RANSAC cutoff as well
    cv::Mat query_img;

    // 1. Decode query image
    if (request_type == "url") {
        std::string image_data_str = download_file(data);
        std::vector<char> image_data_vec(image_data_str.begin(), image_data_str.end());
        query_img = cv::imdecode(image_data_vec, cv::IMREAD_GRAYSCALE);
    } else if (request_type == "image") {
        // Base64 decode
        std::string decoded_data;
        decoded_data.resize(beast::detail::base64::decoded_size(data.size()));
        auto const result = beast::detail::base64::decode(&decoded_data[0], data.data(), data.size());
        decoded_data.resize(result.first);
        std::vector<char> image_data_vec(decoded_data.begin(), decoded_data.end());
        query_img = cv::imdecode(image_data_vec, cv::IMREAD_GRAYSCALE);
    }

    if (query_img.empty()) {
        throw std::runtime_error("Could not decode query image for SIFT processing.");
    }

    // 2. Extract SIFT features from query image
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> query_keypoints;
    cv::Mat query_descriptors;
    sift->detectAndCompute(query_img, cv::noArray(), query_keypoints, query_descriptors);
    
    // *** UPGRADE TO ROOTSIFT ***
    compute_rootsift(query_descriptors);

    if (query_descriptors.empty()) {
        return {}; // No features in query image
    }

    // 3. Process candidates in batches
    std::vector<SiftMatchResult> reranked_results;

    struct CandidateData {
        usearch_key_t key;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };
    std::vector<CandidateData> candidate_features;
    candidate_features.reserve(candidate_keys.size());

    // STAGE 1: SIFT Generation (in parallel)
    START_TIMER(sift_generation_batch);
    #pragma omp parallel
    {
        auto sift_local = cv::SIFT::create();
        std::vector<CandidateData> local_candidate_features;
        #pragma omp for nowait
        for (size_t i = 0; i < candidate_keys.size(); ++i) {
            const auto& key = candidate_keys[i];
            const auto& meta = model_data.metadata_table[key];
            const std::string& uri = meta[uri_column_index];
            
            auto it = model_data.uri_to_file_map.find(uri);
            if (it == model_data.uri_to_file_map.end()) continue;
            
            const std::string& file_path = it->second;
            cv::Mat candidate_img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            if (candidate_img.empty()) continue;

            CandidateData cd;
            cd.key = key;
            sift_local->detectAndCompute(candidate_img, cv::noArray(), cd.keypoints, cd.descriptors);
            if (!cd.descriptors.empty()) {
                // DEBUG: Dump the first candidate's descriptors
                local_candidate_features.push_back(std::move(cd));
            }
        }
        #pragma omp critical
        {
            candidate_features.insert(candidate_features.end(), 
                                      std::make_move_iterator(local_candidate_features.begin()), 
                                      std::make_move_iterator(local_candidate_features.end()));
        }
    }
    END_TIMER(sift_generation_batch);

    // STAGE 2: RootSIFT Conversion (in parallel)
    START_TIMER(rootsift_conversion_batch);
    #pragma omp parallel for
    for(size_t i = 0; i < candidate_features.size(); ++i) {
        compute_rootsift(candidate_features[i].descriptors);
    }
    END_TIMER(rootsift_conversion_batch);

    struct MatchData {
        usearch_key_t key;
        size_t candidate_index;
        std::vector<cv::DMatch> good_matches;
    };
    std::vector<MatchData> good_matches_list;
    good_matches_list.reserve(candidate_features.size());

    // STAGE 3: Brute-Force Matching (in parallel)
    START_TIMER(brute_force_matching_batch);

    good_matches_list.reserve(candidate_features.size());

    #pragma omp parallel
    {
        // Each thread gets its own matcher instance
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false); // No cross-check
        std::vector<MatchData> thread_local_results;

        #pragma omp for nowait schedule(dynamic)
        for (size_t i = 0; i < candidate_features.size(); ++i) {
            const auto& cd = candidate_features[i];
            if (cd.descriptors.empty()) continue;

            // Find 2 nearest neighbors for each query descriptor
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(query_descriptors, cd.descriptors, knn_matches, 2);

            // Apply Lowe's Ratio Test to filter for good matches
            std::vector<cv::DMatch> good_matches;
            const float ratio_thresh = 0.75f;
            for (size_t k = 0; k < knn_matches.size(); ++k) {
                // Ensure we have two matches to compare
                if (knn_matches[k].size() == 2 && knn_matches[k][0].distance < ratio_thresh * knn_matches[k][1].distance) {
                    good_matches.push_back(knn_matches[k][0]);
                }
            }

            // As per proposal, lower the threshold for initial consideration
            if (good_matches.size() >= 10) {
                thread_local_results.push_back({cd.key, i, std::move(good_matches)});
            }
        }

        #pragma omp critical
        {
            good_matches_list.insert(
                good_matches_list.end(),
                std::make_move_iterator(thread_local_results.begin()),
                std::make_move_iterator(thread_local_results.end())
            );
        }
    }
    END_TIMER(brute_force_matching_batch);

    // STAGE 4: Geometric Verification (Voting + Homography)
    START_TIMER(geometric_verification_batch);

    #ifdef DEBUG_SIFT
    // Create a map for quick distance lookup
    std::unordered_map<usearch_key_t, distance_t> key_to_distance;
    for (size_t i = 0; i < candidate_keys.size(); ++i) {
        key_to_distance[candidate_keys[i]] = candidate_distances[i];
    }

    struct SiftDebugData {
        std::string candidate_uri;
        size_t query_kps;
        size_t candidate_kps;
        size_t good_matches;
        int peak_votes;
        int inliers;
        float original_score;
        bool passed_voting;
        bool passed_ransac;
    };
    std::vector<SiftDebugData> debug_data_list;
    #endif

    // Define histogram bins
    const int ROTATION_BINS = 36; // 10 degrees per bin
    const int SCALE_BINS = 10;

    for (const auto& md : good_matches_list) {
        const CandidateData& cd = candidate_features[md.candidate_index];

        // 4a. Hough-style Voting Pre-filter
        cv::Mat votes = cv::Mat::zeros(ROTATION_BINS, SCALE_BINS, CV_32S);
        for (const auto& match : md.good_matches) {
            const auto& q_kp = query_keypoints[match.queryIdx];
            const auto& c_kp = cd.keypoints[match.trainIdx];

            // Calculate rotation difference
            float delta_rotation = q_kp.angle - c_kp.angle;
            if (delta_rotation < 0) delta_rotation += 360.0f;

            // Calculate log-scale difference
            float log_scale = std::log2(q_kp.size / c_kp.size);

            // Map to bins (Rotation: 0-360 -> 0-35, Scale: e.g. -2 to 2 -> 0-9)
            int rotation_bin = static_cast<int>(delta_rotation / 10.0f) % ROTATION_BINS;
            float scale_bin_float = (log_scale + 2.0f) * (SCALE_BINS / 4.0f);
            int scale_bin = std::max(0, std::min(SCALE_BINS - 1, static_cast<int>(scale_bin_float)));
            
            votes.at<int>(rotation_bin, scale_bin)++;
        }

        double max_val;
        cv::minMaxLoc(votes, nullptr, &max_val, nullptr, nullptr);
        int peak_votes = static_cast<int>(max_val);

        bool passed_voting_filter = peak_votes >= MIN_VOTE_THRESHOLD;
        int inlier_count = 0;

        // 4b. Homography/RANSAC for candidates that pass the pre-filter
        if (passed_voting_filter) {
            std::vector<cv::Point2f> query_pts;
            std::vector<cv::Point2f> candidate_pts;
            for (const auto& match : md.good_matches) {
                query_pts.push_back(query_keypoints[match.queryIdx].pt);
                candidate_pts.push_back(cd.keypoints[match.trainIdx].pt);
            }

            if (md.good_matches.size() >= 4) {
                std::vector<uchar> inliers_mask;
                cv::findHomography(query_pts, candidate_pts, cv::USAC_MAGSAC, 3.0, inliers_mask);
                inlier_count = cv::countNonZero(inliers_mask);
            }
        }
        
        bool passed_ransac_filter = inlier_count >= MIN_INLIERS_CUTOFF;

        #ifdef DEBUG_SIFT
        const auto& meta_debug = model_data.metadata_table[md.key];
        const std::string& uri_debug = meta_debug[uri_column_index];
        float original_score = 1.0f - key_to_distance[md.key];
        debug_data_list.push_back({uri_debug, query_keypoints.size(), cd.keypoints.size(), md.good_matches.size(), peak_votes, inlier_count, original_score, passed_voting_filter, passed_ransac_filter});
        #endif

        if (passed_voting_filter && passed_ransac_filter) {
            SiftMatchResult res;
            res.key = md.key;
            res.score = static_cast<float>(inlier_count); // Final score is RANSAC inliers
            reranked_results.push_back(res);
        }
    }
    END_TIMER(geometric_verification_batch);
    
    #ifdef DEBUG_SIFT
    // Sort debug data by inliers, then by votes
    std::sort(debug_data_list.begin(), debug_data_list.end(), [](const auto& a, const auto& b) {
        if (a.inliers != b.inliers) return a.inliers > b.inliers;
        return a.peak_votes > b.peak_votes;
    });

    std::cout << "--- SIFT Reranking Debug ---" << std::endl;
    std::cout << "Query KPs: " << query_keypoints.size() << ", Total Candidates initially: " << candidate_keys.size() << ", Candidates with features: " << candidate_features.size() << std::endl;
    for (const auto& debug_info : debug_data_list) {
        std::cout << "DEBUG SIFT: candidate_uri=\"" << debug_info.candidate_uri
                  << "\", candidate_kps=" << debug_info.candidate_kps
                  << ", good_matches=" << debug_info.good_matches
                  << ", peak_votes=" << debug_info.peak_votes << " (Cutoff: " << MIN_VOTE_THRESHOLD << ", " << (debug_info.passed_voting ? "PASS" : "FAIL") << ")"
                  << ", inliers=" << debug_info.inliers << " (Cutoff: " << MIN_INLIERS_CUTOFF << ", " << (debug_info.passed_ransac ? "PASS" : "FAIL") << ")"
                  << ", final_result=" << ((debug_info.passed_voting && debug_info.passed_ransac) ? "RETURNED" : "DISCARDED")
                  << ", original_score=" << debug_info.original_score << std::endl;
    }
    #endif
    
    // Sort by score descending
    std::sort(reranked_results.begin(), reranked_results.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    return reranked_results;
}


std::string handle_sparql_request(beast::string_view body) {
    START_TIMER(total_request);
    START_TIMER(query_parsing);

    static const std::regex data_regex("(?:<https://artresearch\\.net/embeddings/data>|emb:data)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex request_type_regex("(?:<https://artresearch\\.net/embeddings/request_type>|emb:request_type)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex limit_regex("limit\\s+(\\d+)", std::regex_constants::icase);
    static const std::regex model_regex("(?:<https://artresearch\\.net/embeddings/model>|emb:model)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex joinon_regex("(?:<https://artresearch\\.net/embeddings/joinOn>|emb:joinOn)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex returnvalues_regex("(?:<https://artresearch\\.net/embeddings/returnValues>|emb:returnValues)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex sensitivity_regex("(?:<https://artresearch\\.net/embeddings/sensitivity>|emb:sensitivity)\\s*\"([^\"]*)\"", std::regex_constants::icase);
    static const std::regex exact_regex("(?:<https://artresearch\\.net/embeddings/exact>|emb:exact)\\s*\"(true|false)\"", std::regex_constants::icase);

    std::string query(body);
    std::smatch matches;

    std::string model_names_str;
    if (std::regex_search(query, matches, model_regex)) {
        model_names_str = matches[1].str();
    } else {
        return "{\"error\":\"Model not found in query\"}";
    }

    std::vector<std::string> model_names;
    boost::split(model_names, model_names_str, boost::is_any_of(","));

    if (model_names.empty()) {
        return "{\"error\":\"Model not found in query\"}";
    }

    // Single and Multi-model search logic are now unified.
    std::string data;
    if (std::regex_search(query, matches, data_regex)) {
        data = matches[1].str();
    } else {
        return "{\"error\":\"Data not found in query\"}";
    }

    std::string request_type;
    if (std::regex_search(query, matches, request_type_regex)) {
        request_type = matches[1].str();
    } else {
        return "{\"error\":\"Request type not found in query\"}";
    }

    size_t limit = 10;
    if (std::regex_search(query, matches, limit_regex)) {
        limit = std::stoi(matches[1].str());
    }

    std::string sensitivity;
    if (std::regex_search(query, matches, sensitivity_regex)) {
        sensitivity = matches[1].str();
    }

    bool exact_search = false;
    if (std::regex_search(query, matches, exact_regex)) {
        std::string exact_val = matches[1].str();
        boost::algorithm::to_lower(exact_val);
        if (exact_val == "true") {
            exact_search = true;
        }
    }

    if (request_type != "text" && request_type != "url" && request_type != "image") {
        return "{\"error\":\"Invalid request type for multi-model search\"}";
    }

    std::string return_values_str;
    if (std::regex_search(query, matches, returnvalues_regex)) {
        return_values_str = matches[1].str();
    } else {
        return_values_str = "uri";
    }
    std::vector<std::string> return_values_columns;
    boost::split(return_values_columns, return_values_str, boost::is_any_of(","));

    std::string join_on_str;
    if (std::regex_search(query, matches, joinon_regex)) {
        join_on_str = matches[1].str();
    } else {
        join_on_str = return_values_str; // Default joinOn to returnValues
    }
    std::vector<std::string> join_on_columns;
    boost::split(join_on_columns, join_on_str, boost::is_any_of(","));

    END_TIMER(query_parsing);
    START_TIMER(result_construction);
    // Phase 1: Per-Model Processing
    std::vector<IntermediateResult> all_intermediate_results;
    std::vector<std::string> grouping_models_vec;
    bool is_image_request = (request_type == "url" || request_type == "image");

    // Pre-determine the header variables based on query parameters, not results.
    for (size_t model_idx = 0; model_idx < model_names.size(); ++model_idx) {
        const auto& model_name = model_names[model_idx];
        std::string return_column_name = (model_idx < return_values_columns.size()) ? return_values_columns[model_idx] : return_values_columns.back();
        bool needs_aggregation = (return_column_name != "uri");
        if (needs_aggregation) {
            grouping_models_vec.push_back(model_name);
            if (is_image_request) {
                grouping_models_vec.push_back(model_name + "_sift_rerank");
            }
        }
    }

    for (size_t model_idx = 0; model_idx < model_names.size(); ++model_idx) {
        const auto& model_name = model_names[model_idx];
        auto it = models.find(model_name);
        if (it == models.end()) continue;
        ModelData& model_data = it->second;

        // Determine column indices for this model
        size_t uri_column_index = model_data.metadata_header_map.at("uri");
        std::string join_column_name = (model_idx < join_on_columns.size()) ? join_on_columns[model_idx] : join_on_columns.back();
        std::string return_column_name = (model_idx < return_values_columns.size()) ? return_values_columns[model_idx] : return_values_columns.back();
        
        auto it_join_col = model_data.metadata_header_map.find(join_column_name);
        if (it_join_col == model_data.metadata_header_map.end()) continue;
        size_t join_column_index = it_join_col->second;

        auto it_return_col = model_data.metadata_header_map.find(return_column_name);
        if (it_return_col == model_data.metadata_header_map.end()) continue;
        size_t return_column_index = it_return_col->second;

        // Get embedding
        START_TIMER(embedding_generation);
        std::vector<float> embedding;
        try {
            std::string embedding_payload = data;
            if (request_type != "text") {
                embedding_payload = "{\"type\":\"" + request_type + "\",\"data\":\"" + data + "\"}";
            }
            embedding = get_embedding_from_socket(embedding_payload, model_data);
        } catch (const std::exception& e) {
            std::cerr << "Error getting embedding for model " << model_name << ": " << e.what() << std::endl;
            continue;
        }
        END_TIMER(embedding_generation);

        std::vector<IntermediateResult> current_model_results;

        // Perform embedding search
        START_TIMER(actual_search);
        size_t search_limit = (sensitivity == "near-exact") ? 50 : 1000;
        dense_search_result_t result = model_data.index.search(embedding.data(), search_limit, index_t::any_thread(), exact_search);
        END_TIMER(actual_search);

        if (sensitivity == "near-exact") {
            if (model_names.size() > 1) return "{\"error\":\"near-exact sensitivity can only be used with a single model\"}";
            if (!is_image_request) return "{\"error\":\"near-exact sensitivity can only be used with image requests\"}";
            
            if (result.size() > 0) {
                std::vector<usearch_key_t> found_keys(result.size());
                std::vector<distance_t> found_distances(result.size());
                result.dump_to(found_keys.data(), found_distances.data());
                
                std::vector<SiftMatchResult> raw_sift_results;
                std::string sift_model_name = model_name + "_sift_rerank";
                std::size_t data_hash = std::hash<std::string>{}(data);
                std::string cache_key = model_name + ":" + std::to_string(data_hash); // Key is now just model + data

                if (model_data.sift_cache && model_data.sift_cache->exists(cache_key)) {
                    raw_sift_results = model_data.sift_cache->get(cache_key);
                } else {
                    START_TIMER(sift_reranking);
                    raw_sift_results = rerank_with_sift(data, request_type, found_keys, found_distances, model_data, uri_column_index);
                    END_TIMER(sift_reranking);
                    if (model_data.sift_cache) {
                        model_data.sift_cache->put(cache_key, raw_sift_results);
                    }
                }

                // Now, process the raw results into IntermediateResults
                if (!raw_sift_results.empty()) {
                    float max_sift_score = 0.0f;
                    for(const auto& res : raw_sift_results) max_sift_score = std::max(max_sift_score, res.score);

                    if (max_sift_score > 0.0f) {
                        for (const auto& raw_res : raw_sift_results) {
                            const auto& meta = model_data.metadata_table[raw_res.key];
                            IntermediateResult res;
                            res.group_key = meta[return_column_index];
                            res.join_key = meta[join_column_index];
                            res.score = raw_res.score / max_sift_score; // Normalize
                            res.original_uri = meta[uri_column_index];
                            res.model_name = sift_model_name;
                            current_model_results.push_back(res);
                        }
                    }
                }
            }
        } else {
            // Combined search (SIFT + Embedding) for images, or just embedding for text
            if (result.size() > 0) {
                std::vector<usearch_key_t> found_keys(result.size());
                std::vector<distance_t> found_distances(result.size());
                result.dump_to(found_keys.data(), found_distances.data());

                if (is_image_request) {
                    std::vector<usearch_key_t> keys_for_sift;
                    std::vector<distance_t> distances_for_sift;
                    keys_for_sift.reserve(50);
                    distances_for_sift.reserve(50);

                    for (size_t i = 0; i < std::min((size_t)50, result.size()); ++i) {
                        if (1.0f - found_distances[i] >= 0.75f) {
                            keys_for_sift.push_back(found_keys[i]);
                            distances_for_sift.push_back(found_distances[i]);
                        }
                    }

                    if (!keys_for_sift.empty()) {
                        std::vector<SiftMatchResult> raw_sift_results;
                        std::string sift_model_name = model_name + "_sift_rerank";
                        std::size_t data_hash = std::hash<std::string>{}(data);
                        std::string cache_key = model_name + ":" + std::to_string(data_hash); // Key is now just model + data

                        if (model_data.sift_cache && model_data.sift_cache->exists(cache_key)) {
                            raw_sift_results = model_data.sift_cache->get(cache_key);
                        } else {
                            START_TIMER(sift_reranking);
                            raw_sift_results = rerank_with_sift(data, request_type, keys_for_sift, distances_for_sift, model_data, uri_column_index);
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
                                    res.score = raw_res.score / max_sift_score; // Normalize
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

        // Process and conditionally aggregate results for the current model
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

    // Phase 2: Multi-Model Merging
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
                weight = 1.0f / model_names.size();
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

    // Create a sorted list of results
    std::vector<std::pair<std::string, FinalCombinedResult>> sorted_results(final_results.begin(), final_results.end());
    std::sort(sorted_results.begin(), sorted_results.end(), [](const auto& a, const auto& b) {
        return a.second.final_weighted_score > b.second.final_weighted_score;
    });

    // Phase 3: Sensitivity-based trimming on final scores
    if (!sensitivity.empty() && sensitivity != "near-exact" && sorted_results.size() > 10) { // Only apply if sensitivity is set and we have enough data
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

        float cutoff_score = -1.0f; // Default to not filtering if no non-sift scores
        if (!non_sift_scores.empty()) {
            float sum = std::accumulate(non_sift_scores.begin(), non_sift_scores.end(), 0.0f);
            float mean = sum / non_sift_scores.size();
            float sq_sum = std::inner_product(non_sift_scores.begin(), non_sift_scores.end(), non_sift_scores.begin(), 0.0f);
            float std_dev = std::sqrt(sq_sum / non_sift_scores.size() - mean * mean);

            if (sensitivity == "precise") {
                cutoff_score = mean + sensitivity_factors[0] * std_dev;
            } else if (sensitivity == "balanced") {
                cutoff_score = mean + sensitivity_factors[1] * std_dev;
            } else if (sensitivity == "exploratory") {
                cutoff_score = mean + sensitivity_factors[2] * std_dev;
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
        // Fallback to defaults if the result set is too small
        if (sensitivity == "precise" && new_size < static_cast<size_t>(sensitivity_defaults[0])) new_size = std::min(static_cast<size_t>(sensitivity_defaults[0]), sorted_results.size());
        else if (sensitivity == "balanced" && new_size < static_cast<size_t>(sensitivity_defaults[1])) new_size = std::min(static_cast<size_t>(sensitivity_defaults[1]), sorted_results.size());
        else if (sensitivity == "exploratory" && new_size < static_cast<size_t>(sensitivity_defaults[2])) new_size = std::min(static_cast<size_t>(sensitivity_defaults[2]), sorted_results.size());

        sorted_results = trimmed_results;
        if (sorted_results.size() > new_size) {
            sorted_results.resize(new_size);
        }

    } else if (!sensitivity.empty() && sensitivity != "near-exact") { // Not enough data, use defaults
        size_t default_limit = sorted_results.size();
        if (sensitivity == "precise") default_limit = sensitivity_defaults[0];
        else if (sensitivity == "balanced") default_limit = sensitivity_defaults[1];
        else if (sensitivity == "exploratory") default_limit = sensitivity_defaults[2];
        
        if (sorted_results.size() > default_limit) {
            sorted_results.resize(default_limit);
        }
    }


    END_TIMER(result_construction);
    auto response = format_sparql_response_new(sorted_results, grouping_models_vec);
    END_TIMER(total_request);
    return response;
}

template<class Body, class Allocator>
http::response<http::string_body> handle_request(http::request<Body, http::basic_fields<Allocator>>&& req) {
    auto const bad_request =
    [&req](beast::string_view why) {
        http::response<http::string_body> res{http::status::bad_request, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = std::string(why);
        res.prepare_payload();
        return res;
    };

    auto const not_found =
    [&req](beast::string_view target) {
        http::response<http::string_body> res{http::status::not_found, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "The resource '" + std::string(target) + "' was not found.";
        res.prepare_payload();
        return res;
    };
    

    auto const make_response =
    [&req](beast::string_view body, beast::string_view content_type) {
        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, content_type);
        res.keep_alive(req.keep_alive());
        res.body() = std::string(body);
        res.prepare_payload();
        return res;
    };

    if (req.method() != http::verb::get && req.method() != http::verb::post)
        return bad_request("Unknown HTTP-method");

    boost::system::result<urls::url_view> url_view = urls::parse_origin_form(req.target());
    if (!url_view)
        return bad_request("Cannot parse URL");

    if (url_view->path() == "/sparql") {
        if (req.method() != http::verb::post)
            return bad_request("SPARQL endpoint requires POST");
        return make_response(handle_sparql_request(req.body()), "application/sparql-results+json");
    } else {
        return not_found(req.target());
    }
}

void fail(beast::error_code ec, char const* what) {
    std::cerr << what << ": " << ec.message() << "\n";
}

class session : public std::enable_shared_from_this<session> {
    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    std::optional<http::request_parser<http::string_body>> parser_;

public:
    session(tcp::socket&& socket) : stream_(std::move(socket)) {}

    void run() {
        asio::dispatch(stream_.get_executor(),
                       beast::bind_front_handler(&session::do_read, shared_from_this()));
    }

private:
    void do_read() {
        parser_.emplace();
        parser_->body_limit(10 * 1024 * 1024); // 10MB
        http::async_read(stream_, buffer_, *parser_,
                         beast::bind_front_handler(&session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if (ec == http::error::end_of_stream)
            return do_close();
        if (ec)
            return fail(ec, "read");

        send_response(handle_request(parser_->release()));
    }

    void send_response(http::message_generator&& msg) {
        bool keep_alive = msg.keep_alive();
        beast::async_write(stream_, std::move(msg),
                           beast::bind_front_handler(&session::on_write, shared_from_this(), keep_alive));
    }

    void on_write(bool keep_alive, beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if (ec)
            return fail(ec, "write");
        if (!keep_alive)
            return do_close();
        do_read();
    }

    void do_close() {
        beast::error_code ec;
        stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
    }
};

class listener : public std::enable_shared_from_this<listener> {
    asio::io_context& ioc_;
    tcp::acceptor acceptor_;

public:
    listener(asio::io_context& ioc, tcp::endpoint endpoint)
        : ioc_(ioc), acceptor_(ioc) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec);
        if (ec) {
            fail(ec, "open");
            return;
        }
        acceptor_.set_option(asio::socket_base::reuse_address(true), ec);
        if (ec) {
            fail(ec, "set_option");
            return;
        }
        acceptor_.bind(endpoint, ec);
        if (ec) {
            fail(ec, "bind");
            return;
        }
        acceptor_.listen(asio::socket_base::max_listen_connections, ec);
        if (ec) {
            fail(ec, "listen");
            return;
        }
    }

    void run() {
        do_accept();
    }

private:
    void do_accept() {
        acceptor_.async_accept(
            asio::make_strand(ioc_),
            beast::bind_front_handler(&listener::on_accept, shared_from_this()));
    }

    void on_accept(beast::error_code ec, tcp::socket socket) {
        if (ec) {
            fail(ec, "accept");
        } else {
            std::make_shared<session>(std::move(socket))->run();
        }
        do_accept();
    }
};

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
        ("model-uri-map", po::value<std::vector<std::string>>()->multitoken(), "Pair of model name and path to the CSV mapping URIs to local file paths.");

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

    for (size_t i = 0; i < model_names.size(); ++i) {
        try {
            std::cout << "Loading model " << model_names[i] << "..." << std::endl;
            ModelData model_data;
            model_data.embedding_socket_path = embedding_sockets[i];
            if (cache_size > 0) {
                model_data.embedding_cache = std::make_unique<fifo_cache<std::string, std::vector<float>>>(cache_size);
            }
            // Hardcoded SIFT cache size as requested
            model_data.sift_cache = std::make_unique<fifo_cache<std::string, std::vector<SiftMatchResult>>>(1000);
            std::cout << "Loading index from " << index_files[i] << "..." << std::endl;
            model_data.index.load(index_files[i].c_str());
            std::cout << "Loading metadata from " << metadata_files[i] << "..." << std::endl;
            load_metadata(metadata_files[i], model_data);
            
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
    std::make_shared<listener>(ioc, tcp::endpoint{address, port})->run();

    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads > 1 ? threads - 1 : 0);
    for (auto i = threads - 1; i > 0; --i)
        threads_vec.emplace_back([&ioc] { ioc.run(); });
    
    std::cout << "Server started on port " << port << " with " << threads << " threads" << std::endl;
    ioc.run();

    return 0;
}
