#include "sift_reranker.hpp"
#include "downloader.hpp"
#include <opencv2/opencv.hpp>
#include <boost/beast/core/detail/base64.hpp>
#include <iostream>
#include <fstream>

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

namespace beast = boost::beast;

namespace ignis {
namespace image {

// Function to convert a matrix of SIFT descriptors to RootSIFT
static void compute_rootsift(cv::Mat& descriptors) {
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

    // 3. L2-normalize each descriptor to make Euclidean distance equivalent to Hellinger kernel
    for (int i = 0; i < descriptors.rows; ++i) {
        double norm = cv::norm(descriptors.row(i), cv::NORM_L2);
        if (norm > 0) {
            descriptors.row(i) /= norm;
        }
    }
}

std::vector<search::SiftMatchResult> rerank_with_sift(
    const cv::Mat& query_img_color,
    const std::vector<usearch_key_t>& candidate_keys,
    const std::vector<distance_t>& candidate_distances,
    search::ModelData& model_data,
    size_t uri_column_index
) {
    // --- START: PARAMETERS ---
    const int MIN_RAW_VOTES_THRESHOLD = 8;
    const float MIN_HOUGH_CONSENSUS_RATIO = 0.05f;
    const int ABSOLUTE_MIN_HOUGH_VOTES = 3;
    const float MIN_INLIER_RATIO_OF_GOOD_MATCHES = 0.40f;
    const float MIN_INLIER_RATIO_OF_HOUGH_PEAK = 0.50f;
    const int ABSOLUTE_MIN_INLIERS = 25;
    const int SIFT_FEATURE_CAP = 1024;
    const float LOWES_RATIO_TEST_THRESHOLD = 0.8f;
    const float MIN_INLIER_DENSITY = 0.04f;
    // --- END: PARAMETERS ---
    
    // 1. Convert query image to grayscale for SIFT
    cv::Mat query_img;
    if (query_img_color.channels() == 3) {
        cv::cvtColor(query_img_color, query_img, cv::COLOR_BGR2GRAY);
    } else {
        query_img = query_img_color;
    }

    if (query_img.empty()) {
        throw std::runtime_error("Could not process query image for SIFT (is it empty or invalid?).");
    }

    // 2. Extract SIFT features from query image
    auto sift = cv::SIFT::create(SIFT_FEATURE_CAP);
    std::vector<cv::KeyPoint> query_keypoints;
    cv::Mat query_descriptors;
    sift->detectAndCompute(query_img, cv::noArray(), query_keypoints, query_descriptors);
    
    compute_rootsift(query_descriptors);

    if (query_descriptors.empty()) {
        return {}; // No features in query image
    }

    // 3. Process candidates in batches
    std::vector<search::SiftMatchResult> reranked_results;

    struct CandidateData {
        usearch_key_t key;
        cv::Size image_size;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };
    std::vector<CandidateData> candidate_features;
    candidate_features.reserve(candidate_keys.size());

    // STAGE 1: SIFT Generation (in parallel)
    START_TIMER(sift_generation_batch);
    #pragma omp parallel
    {
        auto sift_local = cv::SIFT::create(SIFT_FEATURE_CAP);
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
            cd.image_size = candidate_img.size();
            sift_local->detectAndCompute(candidate_img, cv::noArray(), cd.keypoints, cd.descriptors);
            if (!cd.descriptors.empty()) {
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

    // STAGE 3 & 4: Per-Candidate Matching and Geometric Verification (Parallelized)
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
        int raw_votes; // This is now good_matches.size()
        int peak_hough_votes;
        int hough_cutoff;
        int inliers;
        int final_inlier_cutoff;
        float density;
        int density_cutoff;
        float original_score;
        bool passed_raw_voting;
        bool passed_hough;
        bool passed_ransac;
    };
    std::vector<SiftDebugData> debug_data_list;
    #endif

    const int ROTATION_BINS = 36;
    const int SCALE_BINS = 10;

    #pragma omp parallel
    {
        #ifdef DEBUG_SIFT
        std::vector<SiftDebugData> local_debug_data_list;
        #endif
        std::vector<search::SiftMatchResult> local_reranked_results;

        #pragma omp for nowait
        for (size_t i = 0; i < candidate_features.size(); ++i) {
            const CandidateData& cd = candidate_features[i];
            usearch_key_t key = cd.key;

            if (cd.descriptors.empty()) {
                continue;
            }

            // A. Perform k-NN match (k=2) in both directions for symmetric check.
            auto matcher = cv::BFMatcher::create(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knn_matches_q_to_c;
            matcher->knnMatch(query_descriptors, cd.descriptors, knn_matches_q_to_c, 2);
            
            std::vector<std::vector<cv::DMatch>> knn_matches_c_to_q;
            matcher->knnMatch(cd.descriptors, query_descriptors, knn_matches_c_to_q, 2);

            // B. Apply Lowe's Ratio Test and Mutual NN Check.
            std::vector<cv::DMatch> good_matches;
            std::vector<bool> c_kp_matched(cd.keypoints.size(), false);

            for (const auto& match_pair : knn_matches_q_to_c) {
                if (match_pair.size() == 2 && match_pair[0].distance < LOWES_RATIO_TEST_THRESHOLD * match_pair[1].distance) {
                    const auto& forward_match = match_pair[0];
                    
                    // Check if the reverse match is also good
                    const auto& reverse_match_pair = knn_matches_c_to_q[forward_match.trainIdx];
                    if (reverse_match_pair.size() == 2 && 
                        reverse_match_pair[0].distance < LOWES_RATIO_TEST_THRESHOLD * reverse_match_pair[1].distance) {
                        
                        // Mutual NN check: is the best reverse match the original query keypoint?
                        if (reverse_match_pair[0].trainIdx == forward_match.queryIdx) {
                            // To ensure one-to-one mapping, check if candidate kp is already matched
                            if (!c_kp_matched[forward_match.trainIdx]) {
                                good_matches.push_back(forward_match);
                                c_kp_matched[forward_match.trainIdx] = true;
                            }
                        }
                    }
                }
            }
            
            int good_match_count = good_matches.size();
            bool passed_raw_voting = good_match_count >= MIN_RAW_VOTES_THRESHOLD;
            if (!passed_raw_voting) {
                continue;
            }

            // C. Hough-style Voting Pre-filter
            cv::Mat votes = cv::Mat::zeros(ROTATION_BINS, SCALE_BINS, CV_32S);
            for (const auto& match : good_matches) {
                const auto& q_kp = query_keypoints[match.queryIdx];
                const auto& c_kp = cd.keypoints[match.trainIdx];

                float delta_rotation = q_kp.angle - c_kp.angle;
                if (delta_rotation < 0) delta_rotation += 360.0f;
                float log_scale = std::log2(q_kp.size / c_kp.size);

                int rotation_bin = static_cast<int>(delta_rotation / 10.0f) % ROTATION_BINS;
                float scale_bin_float = (log_scale + 2.0f) * (SCALE_BINS / 4.0f);
                int scale_bin = std::max(0, std::min(SCALE_BINS - 1, static_cast<int>(scale_bin_float)));
                
                votes.at<int>(rotation_bin, scale_bin)++;
            }

            double max_val;
            cv::minMaxLoc(votes, nullptr, &max_val, nullptr, nullptr);
            int peak_hough_votes = static_cast<int>(max_val);

            int hough_cutoff = std::max(ABSOLUTE_MIN_HOUGH_VOTES, 
                                        static_cast<int>(good_match_count * MIN_HOUGH_CONSENSUS_RATIO));
            bool passed_hough_filter = peak_hough_votes >= hough_cutoff;
            int inlier_count = 0;

            if (passed_hough_filter) {
                std::vector<cv::Point2f> query_pts;
                std::vector<cv::Point2f> candidate_pts;
                for (const auto& match : good_matches) {
                    query_pts.push_back(query_keypoints[match.queryIdx].pt);
                    candidate_pts.push_back(cd.keypoints[match.trainIdx].pt);
                }

                if (good_matches.size() >= 4) {
                    std::vector<uchar> inliers_mask;
                    float diag = std::sqrt(static_cast<float>(cd.image_size.width * cd.image_size.width + cd.image_size.height * cd.image_size.height));
                    float ransac_thresh = std::max(1.0f, 0.003f * diag);
                    cv::findHomography(query_pts, candidate_pts, cv::USAC_MAGSAC, ransac_thresh, inliers_mask);
                    inlier_count = cv::countNonZero(inliers_mask);
                }
            }

            int inlier_cutoff_from_good = static_cast<int>(good_match_count * MIN_INLIER_RATIO_OF_GOOD_MATCHES);
            int inlier_cutoff_from_hough = static_cast<int>(peak_hough_votes * MIN_INLIER_RATIO_OF_HOUGH_PEAK);
            size_t min_kps = std::min(query_keypoints.size(), cd.keypoints.size());
            int density_cutoff = static_cast<int>(MIN_INLIER_DENSITY * static_cast<float>(min_kps));
            int final_inlier_cutoff = std::max({ABSOLUTE_MIN_INLIERS, inlier_cutoff_from_good, inlier_cutoff_from_hough, density_cutoff});
            bool passed_ransac_filter = inlier_count >= final_inlier_cutoff;

            #ifdef DEBUG_SIFT
            const auto& meta_debug = model_data.metadata_table.at(key);
            const std::string& uri_debug = meta_debug[uri_column_index];
            float original_score = 1.0f - key_to_distance.at(key);
            float density = (min_kps > 0) ? static_cast<float>(inlier_count) / min_kps : 0.0f;
            local_debug_data_list.push_back({uri_debug, query_keypoints.size(), cd.keypoints.size(), good_match_count, peak_hough_votes, hough_cutoff, inlier_count, final_inlier_cutoff, density, density_cutoff, original_score, passed_raw_voting, passed_hough_filter, passed_ransac_filter});
            #endif

            if (passed_hough_filter && passed_ransac_filter) {
                search::SiftMatchResult res;
                res.key = key;
                size_t min_kps = std::min(query_keypoints.size(), cd.keypoints.size());
                res.score = (min_kps > 0) ? static_cast<float>(inlier_count) / min_kps : 0.0f;
                local_reranked_results.push_back(res);
            }
        }

        #pragma omp critical
        {
            reranked_results.insert(reranked_results.end(), local_reranked_results.begin(), local_reranked_results.end());
            #ifdef DEBUG_SIFT
            debug_data_list.insert(debug_data_list.end(), local_debug_data_list.begin(), local_debug_data_list.end());
            #endif
        }
    }
    END_TIMER(geometric_verification_batch);
    
    #ifdef DEBUG_SIFT
    // Sort debug data for clarity
    std::sort(debug_data_list.begin(), debug_data_list.end(), [](const auto& a, const auto& b) {
        if (a.inliers != b.inliers) return a.inliers > b.inliers;
        if (a.peak_hough_votes != b.peak_hough_votes) return a.peak_hough_votes > b.peak_hough_votes;
        return a.raw_votes > b.raw_votes;
    });

    std::cout << "--- SIFT Reranking Debug ---" << std::endl;
    std::cout << "Query KPs: " << query_keypoints.size() << ", Total Candidates initially: " << candidate_keys.size() << ", Candidates with features: " << candidate_features.size() << std::endl;
    for (const auto& debug_info : debug_data_list) {
        std::cout << "DEBUG SIFT: candidate_uri=\"" << debug_info.candidate_uri
                  << "\", candidate_kps=" << debug_info.candidate_kps
                  << ", raw_votes=" << debug_info.raw_votes << " (Cutoff: " << MIN_RAW_VOTES_THRESHOLD << ", " << (debug_info.passed_raw_voting ? "PASS" : "FAIL") << ")"
                  << ", hough_votes=" << debug_info.peak_hough_votes << " (Cutoff: " << debug_info.hough_cutoff << ", " << (debug_info.passed_hough ? "PASS" : "FAIL") << ")"
                  << ", inliers=" << debug_info.inliers << " (Cutoff: " << debug_info.final_inlier_cutoff << ", " << (debug_info.passed_ransac ? "PASS" : "FAIL") << ")"
                  << ", density=" << debug_info.density << " (Cutoff: " << debug_info.density_cutoff << ", " << (debug_info.density >= MIN_INLIER_DENSITY ? "PASS" : "FAIL") << ")"
                  << ", final_result=" << ((debug_info.passed_hough && debug_info.passed_ransac) ? "RETURNED" : "DISCARDED")
                  << ", original_score=" << debug_info.original_score << std::endl;
    }
    #endif
    
    std::sort(reranked_results.begin(), reranked_results.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    return reranked_results;
}

} // namespace image
} // namespace ignis
