#ifndef IGNIS_IMAGE_DOWNLOADER_HPP
#define IGNIS_IMAGE_DOWNLOADER_HPP

#include <string>

namespace ignis {
namespace image {

/**
 * @brief Downloads a file from a given URL.
 *
 * This function uses cURL to download the content of a URL into a string.
 * It sets a user agent and follows redirects.
 *
 * @param url The URL of the file to download.
 * @return A string containing the downloaded file data.
 * @throws std::runtime_error if cURL fails to initialize or the download fails.
 */
std::string download_file(const std::string& url);

} // namespace image
} // namespace ignis

#endif // IGNIS_IMAGE_DOWNLOADER_HPP
