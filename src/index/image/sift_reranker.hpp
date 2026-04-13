#ifndef IGNIS_IMAGE_SIFT_RERANKER_HPP
#define IGNIS_IMAGE_SIFT_RERANKER_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp> // Add OpenCV header
#include "search/model.hpp"

namespace ignis {
namespace image {

/**
 * @brief Reranks a list of candidate keys using SIFT feature matching.
 *
 * This function takes a query image as a cv::Mat and a list of
 * candidate keys from an initial search. It then performs SIFT feature matching
 * between the query image and the images corresponding to the candidate keys.
 * The results are reranked based on the number of inliers from a homography check.
 *
 * @param query_img The query image, already decoded into a cv::Mat.
 * @param candidate_keys A vector of candidate keys to rerank.
 * @param candidate_distances The original distances of the candidates.
 * @param model_data The ModelData for the current model.
 * @param uri_column_index The index of the URI column in the metadata.
 * @return A vector of SiftMatchResult, sorted by score in descending order.
 */
std::vector<search::SiftMatchResult> rerank_with_sift(
    const cv::Mat& query_img,
    const std::vector<usearch_key_t>& candidate_keys,
    const std::vector<distance_t>& candidate_distances,
    search::ModelData& model_data,
    size_t uri_column_index
);

} // namespace image
} // namespace ignis

#endif // IGNIS_IMAGE_SIFT_RERANKER_HPP
