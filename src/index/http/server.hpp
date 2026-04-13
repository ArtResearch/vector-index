#ifndef IGNIS_HTTP_SERVER_HPP
#define IGNIS_HTTP_SERVER_HPP

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <string>

namespace ignis {
namespace http {

namespace beast = boost::beast;
namespace http = beast::http;
namespace asio = boost::asio;
using tcp = asio::ip::tcp;

/**
 * @brief Handles an incoming HTTP request and produces a response.
 *
 * This is a function object that will be passed to the server session.
 */
using RequestHandler = std::function<http::response<http::string_body>(http::request<http::string_body>&&)>;

/**
 * @brief Accepts incoming connections and creates sessions.
 */
class listener : public std::enable_shared_from_this<listener> {
public:
    listener(asio::io_context& ioc, tcp::endpoint endpoint, RequestHandler handler);
    void run();

private:
    void do_accept();
    void on_accept(beast::error_code ec, tcp::socket socket);

    asio::io_context& _ioc;
    tcp::acceptor _acceptor;
    RequestHandler _handler;
};

} // namespace http
} // namespace ignis

#endif // IGNIS_HTTP_SERVER_HPP
