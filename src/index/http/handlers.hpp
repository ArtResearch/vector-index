#ifndef IGNIS_HTTP_HANDLERS_HPP
#define IGNIS_HTTP_HANDLERS_HPP

#include "server.hpp"
#include "search/service.hpp"
#include <memory>

namespace ignis {
namespace http {

/**
 * @brief Creates a request handler for the search server.
 *
 * This function returns a lambda that captures the search service and handles
 * incoming HTTP requests. It routes requests to the appropriate service methods
 * and formats the responses.
 *
 * @param service A shared pointer to the search service.
 * @return A RequestHandler function object.
 */
RequestHandler create_handler(std::shared_ptr<search::Service> service);

} // namespace http
} // namespace ignis

#endif // IGNIS_HTTP_HANDLERS_HPP
