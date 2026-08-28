#include "http.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <signal.h>

namespace http {
void Response::write_all(const std::string& s) { size_t off = 0; while (off < s.size()) { ssize_t n = ::send(fd_, s.data() + off, s.size() - off, MSG_NOSIGNAL); if (n <= 0) throw std::runtime_error("client gone"); off += n; } }
void Response::send(int status, const std::string& type, const std::string& body) {
    if (done_) return;
    std::string h = "HTTP/1.1 " + std::to_string(status) + (status == 200 ? " OK" : status == 400 ? " Bad Request" : status == 404 ? " Not Found" : " Error") + "\r\nContent-Type: " + type + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    write_all(h + body); done_ = true;
}
void Response::begin_stream(const std::string& type) {
    write_all("HTTP/1.1 200 OK\r\nContent-Type: " + type + "\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"); streaming_ = true;
}
void Response::chunk(const std::string& data) { char b[32]; snprintf(b, sizeof b, "%zx\r\n", data.size()); write_all(b + data + "\r\n"); }
void Response::end() { if (streaming_ && !done_) { write_all("0\r\n\r\n"); done_ = true; } }

void serve(const std::string& host, int port, const Handler& handler) {
    signal(SIGPIPE, SIG_IGN);
    int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    if (bind(s, (sockaddr*)&a, sizeof a) < 0 || listen(s, 16) < 0) throw std::runtime_error("bind/listen failed on port " + std::to_string(port));
    fprintf(stderr, "listening on http://%s:%d\n", host.c_str(), port);
    while (true) {
        int fd = accept(s, nullptr, nullptr); if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::string buf; char tmp[65536]; size_t hdr_end = std::string::npos;
        while (hdr_end == std::string::npos) { ssize_t n = recv(fd, tmp, sizeof tmp, 0); if (n <= 0) break; buf.append(tmp, n); hdr_end = buf.find("\r\n\r\n"); if (buf.size() > (64u << 20)) break; }
        if (hdr_end == std::string::npos) { close(fd); continue; }
        Request r; { size_t p = buf.find(' '); r.method = buf.substr(0, p); size_t q = buf.find(' ', p + 1); r.path = buf.substr(p + 1, q - p - 1); }
        size_t line = buf.find("\r\n") + 2;
        while (line < hdr_end) { size_t e = buf.find("\r\n", line); size_t c = buf.find(':', line); if (c != std::string::npos && c < e) { std::string k = buf.substr(line, c - line), v = buf.substr(c + 1, e - c - 1); for (auto& ch : k) ch = tolower(ch); while (!v.empty() && v[0] == ' ') v.erase(0, 1); r.headers[k] = v; } line = e + 2; }
        size_t clen = r.headers.count("content-length") ? std::stoul(r.headers["content-length"]) : 0;
        r.body = buf.substr(hdr_end + 4);
        while (r.body.size() < clen) { ssize_t n = recv(fd, tmp, sizeof tmp, 0); if (n <= 0) break; r.body.append(tmp, n); }
        Response resp(fd);
        try { handler(r, resp); } catch (const std::exception& e) { try { if (!resp.streaming()) resp.send(500, "application/json", std::string("{\"error\":{\"message\":\"") + e.what() + "\"}}"); else { resp.chunk("data: {\"error\":\"" + std::string(e.what()) + "\"}\n\n"); resp.end(); } } catch (...) {} }
        try { resp.end(); } catch (...) {}
        close(fd);
    }
}
}  // namespace http
