#include "client.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <arpa/inet.h> // For htonl
#include <cerrno>
#include <sys/time.h>
#include <cstring> // For memset
#include <cstddef> // For offsetof

namespace ignis {
namespace embedding {

// Helper function to ensure all data is sent over the socket
static void send_all(int sock, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t total_sent = 0;
    while (total_sent < len) {
        ssize_t sent = send(sock, p + total_sent, len - total_sent, MSG_NOSIGNAL);
        if (sent == -1) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            throw std::runtime_error("Failed to send data over socket: " + std::string(strerror(errno)));
        }
        total_sent += sent;
    }
}

// Helper function to ensure all data is read from the socket
static void read_exact(int sock, void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    size_t total_read = 0;
    while (total_read < len) {
        ssize_t result = read(sock, p + total_read, len - total_read);
        if (result == -1) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            throw std::runtime_error("Failed to read data from socket: " + std::string(strerror(errno)));
        }
        if (result == 0) {
            throw std::runtime_error("Socket closed prematurely while reading data");
        }
        total_read += result;
    }
}

// Forward declaration for the new binary protocol implementation
std::vector<float> get_embedding_from_socket(const std::string& payload, PayloadType type, search::ModelData& model_data);

// Keep the old function signature for compatibility, defaulting to Text payload
std::vector<float> get_embedding_from_socket(const std::string& payload, search::ModelData& model_data) {
    return get_embedding_from_socket(payload, PayloadType::Text, model_data);
}

// New implementation with binary protocol support
std::vector<float> get_embedding_from_socket(const std::string& payload, PayloadType type, search::ModelData& model_data) {
    // Caching logic
    if (model_data.embedding_cache) {
        if (model_data.embedding_cache->exists(payload)) {
            return model_data.embedding_cache->get(payload);
        }
    }

    int sock = 0;
    if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
        throw std::runtime_error("Socket creation error: " + std::string(strerror(errno)));
    }

    struct timeval tv;
    tv.tv_sec = 10; // 10 second timeout
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);

    struct sockaddr_un serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sun_family = AF_UNIX;
    if (model_data.embedding_socket_path.length() >= sizeof(serv_addr.sun_path)) {
        close(sock);
        throw std::runtime_error("Socket path is too long: " + model_data.embedding_socket_path);
    }
    strncpy(serv_addr.sun_path, model_data.embedding_socket_path.c_str(), sizeof(serv_addr.sun_path) -1);

    socklen_t addrlen = offsetof(struct sockaddr_un, sun_path) + model_data.embedding_socket_path.length();

    if (connect(sock, (struct sockaddr *)&serv_addr, addrlen) < 0) {
        close(sock);
        throw std::runtime_error("Connection failed to embedding server at " + model_data.embedding_socket_path + ": " + strerror(errno));
    }

    // --- Robust Binary Protocol ---
    // 1. Send message type (1 byte)
    uint8_t msg_type = static_cast<uint8_t>(type);
    send_all(sock, &msg_type, 1);

    // 2. Send payload length (4 bytes)
    uint32_t len = htonl(static_cast<uint32_t>(payload.length()));
    send_all(sock, &len, sizeof(len));

    // 3. Send payload
    send_all(sock, payload.data(), payload.length());
    // ---------------------------

    // Receive the response
    uint32_t embedding_size_net;
    read_exact(sock, &embedding_size_net, sizeof(embedding_size_net));
    uint32_t embedding_size = ntohl(embedding_size_net);

    if (embedding_size == 0) {
        close(sock);
        throw std::runtime_error("Embedding server returned an error (embedding size is 0)");
    }

    std::vector<float> embedding(embedding_size);
    read_exact(sock, embedding.data(), embedding_size * sizeof(float));

    close(sock);

    // Cache the new embedding
    if (model_data.embedding_cache) {
        model_data.embedding_cache->put(payload, embedding);
    }

    return embedding;
}

} // namespace embedding
} // namespace ignis
