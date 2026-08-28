// Minimal blocking HTTP/1.1 server: one request at a time (the engine is one slot), Content-Length bodies, plain or
// chunked (SSE) responses. No dependencies.
#pragma once
#include <string>
#include <map>
#include <functional>

namespace http {
struct Request { std::string method, path, body; std::map<std::string, std::string> headers; };
class Response {   // either send(status, type, body) once, or begin_stream() then chunk()... end()
public:
    explicit Response(int fd) : fd_(fd) {}
    void send(int status, const std::string& type, const std::string& body);
    void begin_stream(const std::string& type = "text/event-stream");
    void chunk(const std::string& data);
    void end();
    bool streaming() const { return streaming_; }
private:
    int fd_; bool streaming_ = false; bool done_ = false;
    void write_all(const std::string& s);
};
using Handler = std::function<void(const Request&, Response&)>;
void serve(const std::string& host, int port, const Handler& h);   // blocks forever
}  // namespace http
