#include "server.hpp"
#include <iostream>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>

namespace ignis {
namespace http {

// Report a failure
static void fail(beast::error_code ec, char const* what) {
    std::cerr << what << ": " << ec.message() << "\n";
}

class session : public std::enable_shared_from_this<session> {
    beast::tcp_stream _stream;
    beast::flat_buffer _buffer;
    std::optional<http::request_parser<http::string_body>> _parser;
    RequestHandler _handler;

public:
    session(tcp::socket&& socket, RequestHandler handler)
        : _stream(std::move(socket)), _handler(std::move(handler)) {}

    void run() {
        asio::dispatch(_stream.get_executor(),
                       beast::bind_front_handler(&session::do_read, shared_from_this()));
    }

private:
    void do_read() {
        _parser.emplace();
        _parser->body_limit(50 * 1024 * 1024); // 50MB
        http::async_read(_stream, _buffer, *_parser,
                         beast::bind_front_handler(&session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if (ec == http::error::end_of_stream)
            return do_close();
        if (ec)
            return fail(ec, "read");

        send_response(_handler(_parser->release()));
    }

    void send_response(http::message_generator&& msg) {
        bool keep_alive = msg.keep_alive();
        beast::async_write(_stream, std::move(msg),
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
        _stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }
};

listener::listener(asio::io_context& ioc, tcp::endpoint endpoint, RequestHandler handler)
    : _ioc(ioc), _acceptor(ioc), _handler(std::move(handler)) {
    beast::error_code ec;
    _acceptor.open(endpoint.protocol(), ec);
    if (ec) {
        fail(ec, "open");
        return;
    }
    _acceptor.set_option(asio::socket_base::reuse_address(true), ec);
    if (ec) {
        fail(ec, "set_option");
        return;
    }
    _acceptor.bind(endpoint, ec);
    if (ec) {
        fail(ec, "bind");
        return;
    }
    _acceptor.listen(asio::socket_base::max_listen_connections, ec);
    if (ec) {
        fail(ec, "listen");
        return;
    }
}

void listener::run() {
    do_accept();
}

void listener::do_accept() {
    _acceptor.async_accept(
        asio::make_strand(_ioc),
        beast::bind_front_handler(&listener::on_accept, shared_from_this()));
}

void listener::on_accept(beast::error_code ec, tcp::socket socket) {
    if (ec) {
        fail(ec, "accept");
    } else {
        std::make_shared<session>(std::move(socket), _handler)->run();
    }
    do_accept();
}

} // namespace http
} // namespace ignis
