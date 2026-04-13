#include "handlers.hpp"
#include "sparql/parser.hpp"
#include "sparql/response_formatter.hpp"
#include <boost/url.hpp>
#include <chrono>
#include <iostream>

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

namespace ignis {
namespace http {

namespace urls = boost::urls;

static http::response<http::string_body> bad_request(const http::request<http::string_body>& req, beast::string_view why) {
    http::response<http::string_body> res{http::status::bad_request, req.version()};
    res.set(http::field::server, "Ignis Server");
    res.set(http::field::content_type, "application/json");
    res.keep_alive(req.keep_alive());
    res.body() = "{\"error\":\"" + std::string(why) + "\"}";
    res.prepare_payload();
    return res;
};

static http::response<http::string_body> not_found(const http::request<http::string_body>& req, beast::string_view target) {
    http::response<http::string_body> res{http::status::not_found, req.version()};
    res.set(http::field::server, "Ignis Server");
    res.set(http::field::content_type, "application/json");
    res.keep_alive(req.keep_alive());
    res.body() = "{\"error\":\"The resource '" + std::string(target) + "' was not found.\"}";
    res.prepare_payload();
    return res;
};

static http::response<http::string_body> server_error(const http::request<http::string_body>& req, beast::string_view what) {
    http::response<http::string_body> res{http::status::internal_server_error, req.version()};
    res.set(http::field::server, "Ignis Server");
    res.set(http::field::content_type, "application/json");
    res.keep_alive(req.keep_alive());
    res.body() = "{\"error\":\"" + std::string(what) + "\"}";
    res.prepare_payload();
    return res;
};

static http::response<http::string_body> make_response(const http::request<http::string_body>& req, beast::string_view body, beast::string_view content_type) {
    http::response<http::string_body> res{http::status::ok, req.version()};
    res.set(http::field::server, "Ignis Server");
    res.set(http::field::content_type, content_type);
    res.keep_alive(req.keep_alive());
    res.body() = std::string(body);
    res.prepare_payload();
    return res;
};

RequestHandler create_handler(std::shared_ptr<search::Service> service) {
    return [service](http::request<http::string_body>&& req) -> http::response<http::string_body> {
        START_TIMER(total_request);
        if (req.method() != http::verb::post)
            return bad_request(req, "Unknown HTTP-method");

        boost::system::result<urls::url_view> url_view = urls::parse_origin_form(req.target());
        if (!url_view)
            return bad_request(req, "Cannot parse URL");

        if (url_view->path() == "/sparql") {
            START_TIMER(query_parsing);
            auto query_opt = sparql::parse_sparql_request(req.body());
            if (!query_opt) {
                return bad_request(req, "Invalid SPARQL query parameters");
            }
            END_TIMER(query_parsing);

            try {
                auto [results, grouping_models] = service->search(query_opt.value());
                auto response_body = sparql::format_sparql_response(results, grouping_models);
                auto response = make_response(req, response_body, "application/sparql-results+json");
                END_TIMER(total_request);
                return response;
            } catch (const std::exception& e) {
                return server_error(req, e.what());
            }
        }

        return not_found(req, req.target());
    };
}

} // namespace http
} // namespace ignis
