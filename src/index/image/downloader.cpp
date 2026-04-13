#include "downloader.hpp"
#include <curl/curl.h>
#include <stdexcept>
#include <memory>

namespace ignis {
namespace image {

// Callback function to write received data into a string
static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t new_length = size * nmemb;
    try {
        s->append((char*)contents, new_length);
    } catch (std::bad_alloc& e) {
        // handle memory problem
        return 0;
    }
    return new_length;
}

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

} // namespace image
} // namespace ignis
