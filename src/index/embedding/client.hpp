#ifndef IGNIS_EMBEDDING_CLIENT_HPP
#define IGNIS_EMBEDDING_CLIENT_HPP

#include <string>
#include <vector>
#include "search/model.hpp"

namespace ignis {
namespace embedding {

enum class PayloadType : uint8_t {
    Text = 0x01,
    Image = 0x02
};

/**
 * @brief Fetches a vector embedding from the embedding service via a Unix socket for a text payload.
 *
 * @param payload The text data to be sent to the embedding service.
 * @param model_data The ModelData object containing the socket path and cache.
 * @return A vector of floats representing the embedding.
 */
std::vector<float> get_embedding_from_socket(const std::string& payload, search::ModelData& model_data);

/**
 * @brief Fetches a vector embedding from the embedding service via a Unix socket using the binary protocol.
 *
 * @param payload The data (text or image bytes) to be sent.
 * @param type The type of the payload (Text or Image).
 * @param model_data The ModelData object containing the socket path and cache.
 * @return A vector of floats representing the embedding.
 */
std::vector<float> get_embedding_from_socket(const std::string& payload, PayloadType type, search::ModelData& model_data);

} // namespace embedding
} // namespace ignis

#endif // IGNIS_EMBEDDING_CLIENT_HPP
