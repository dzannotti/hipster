#include "gguf.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <regex>

namespace hip {

const char* gtype_name(GType t) {
    switch (t) {
        case GType::F32: return "F32"; case GType::F16: return "F16"; case GType::BF16: return "BF16";
        case GType::Q4_0: return "Q4_0"; case GType::Q4_1: return "Q4_1"; case GType::Q5_0: return "Q5_0";
        case GType::Q5_1: return "Q5_1"; case GType::Q8_0: return "Q8_0"; case GType::Q8_1: return "Q8_1";
        case GType::Q2_K: return "Q2_K"; case GType::Q3_K: return "Q3_K"; case GType::Q4_K: return "Q4_K";
        case GType::Q5_K: return "Q5_K"; case GType::Q6_K: return "Q6_K"; case GType::Q8_K: return "Q8_K";
        case GType::IQ4_NL: return "IQ4_NL"; case GType::IQ4_XS: return "IQ4_XS"; case GType::IQ3_S: return "IQ3_S";
        case GType::I32: return "I32"; case GType::I8: return "I8";
        default: return "?";
    }
}

std::pair<uint32_t, uint32_t> gtype_block(GType t) {
    switch (t) {
        case GType::F32: return {1, 4}; case GType::F16: return {1, 2}; case GType::BF16: return {1, 2};
        case GType::I8: return {1, 1}; case GType::I32: return {1, 4};
        case GType::Q4_0: return {32, 18}; case GType::Q4_1: return {32, 20};
        case GType::Q5_0: return {32, 22}; case GType::Q5_1: return {32, 24};
        case GType::Q8_0: return {32, 34}; case GType::Q8_1: return {32, 36};
        case GType::Q2_K: return {256, 84}; case GType::Q3_K: return {256, 110};
        case GType::Q4_K: return {256, 144}; case GType::Q5_K: return {256, 176};
        case GType::Q6_K: return {256, 210}; case GType::Q8_K: return {256, 292};
        case GType::IQ4_NL: return {32, 18}; case GType::IQ4_XS: return {256, 136};
        case GType::IQ3_S: return {256, 110};
        default: throw std::runtime_error("gtype_block: unsupported type " + std::to_string((int)t));
    }
}

namespace {
struct Cursor {
    const uint8_t* p; const uint8_t* end;
    template <class T> T rd() { if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated"); T v; memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
    std::string str() { uint64_t n = rd<uint64_t>(); if (p + n > end) throw std::runtime_error("gguf: truncated string"); std::string s((const char*)p, n); p += n; return s; }
};
GValue read_value(Cursor& c, uint32_t type) {
    GValue v;
    switch (type) {
        case 0: v.kind = GValue::U32; v.u = c.rd<uint8_t>(); break;
        case 1: v.kind = GValue::I32; v.i = c.rd<int8_t>(); break;
        case 2: v.kind = GValue::U32; v.u = c.rd<uint16_t>(); break;
        case 3: v.kind = GValue::I32; v.i = c.rd<int16_t>(); break;
        case 4: v.kind = GValue::U32; v.u = c.rd<uint32_t>(); break;
        case 5: v.kind = GValue::I32; v.i = c.rd<int32_t>(); break;
        case 6: v.kind = GValue::F32; v.f = c.rd<float>(); break;
        case 7: v.kind = GValue::BOOL; v.u = c.rd<uint8_t>(); break;
        case 8: v.kind = GValue::STR; v.s = c.str(); break;
        case 9: { v.kind = GValue::ARR; uint32_t et = c.rd<uint32_t>(); uint64_t n = c.rd<uint64_t>();
                  v.arr.reserve(n < (1u << 20) ? n : 0); for (uint64_t k = 0; k < n; ++k) v.arr.push_back(read_value(c, et)); break; }
        case 10: v.kind = GValue::U64; v.u = c.rd<uint64_t>(); break;
        case 11: v.kind = GValue::I64; v.i = c.rd<int64_t>(); break;
        case 12: v.kind = GValue::F64; v.f = c.rd<double>(); break;
        default: throw std::runtime_error("gguf: bad kv type " + std::to_string(type));
    }
    return v;
}
}  // namespace

GGUF::GGUF(const std::string& path) {
    // split naming: <base>-00001-of-0000N.gguf
    static const std::regex re(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (std::regex_match(path, m, re)) {
        int n = std::stoi(m[3]);
        for (int i = 1; i <= n; ++i) {
            char buf[32]; snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, n);
            open_one(m[1].str() + buf);
        }
    } else {
        open_one(path);
    }
}

GGUF::~GGUF() { for (auto& m : maps_) { munmap(m.p, m.len); ::close(m.fd); } }

void GGUF::open_one(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st; fstat(fd, &st);
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("gguf: mmap failed " + path);
    maps_.push_back({p, (size_t)st.st_size, fd});   // fd kept open for posix_fadvise
    mapped_bytes_ += st.st_size;

    Cursor c{(const uint8_t*)p, (const uint8_t*)p + st.st_size};
    if (c.rd<uint32_t>() != 0x46554747) throw std::runtime_error("gguf: bad magic " + path);
    uint32_t ver = c.rd<uint32_t>(); if (ver != 3) throw std::runtime_error("gguf: version != 3");
    uint64_t n_tensors = c.rd<uint64_t>(), n_kv = c.rd<uint64_t>();
    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key = c.str(); uint32_t type = c.rd<uint32_t>();
        GValue v = read_value(c, type);
        if (!kv_.count(key) || key.rfind("split.", 0) != 0) kv_[key] = std::move(v);
    }
    struct TmpT { GTensor t; uint64_t off; };
    std::vector<TmpT> tmp; tmp.reserve(n_tensors);
    for (uint64_t k = 0; k < n_tensors; ++k) {
        TmpT tt; tt.t.name = c.str(); tt.t.n_dims = c.rd<uint32_t>();
        if (tt.t.n_dims > 4) throw std::runtime_error("gguf: n_dims > 4");
        for (uint32_t d = 0; d < tt.t.n_dims; ++d) tt.t.ne[d] = c.rd<uint64_t>();
        tt.t.type = (GType)c.rd<uint32_t>(); tt.off = c.rd<uint64_t>();
        tmp.push_back(std::move(tt));
    }
    uint32_t align = 32;
    if (auto* a = kv("general.alignment")) align = (uint32_t)a->u;
    size_t data_start = ((size_t)(c.p - (const uint8_t*)p) + align - 1) / align * align;
    for (auto& tt : tmp) {
        auto [bs, tb] = gtype_block(tt.t.type);
        uint64_t n = tt.t.ne[0] * tt.t.ne[1] * tt.t.ne[2] * tt.t.ne[3];
        tt.t.nbytes = n / bs * tb;
        tt.t.data = (const uint8_t*)p + data_start + tt.off; tt.t.map_index = (int)maps_.size() - 1; tt.t.file_off = data_start + tt.off;
        if (tt.t.data + tt.t.nbytes > (const uint8_t*)p + st.st_size) throw std::runtime_error("gguf: tensor past EOF " + tt.t.name);
        index_[tt.t.name] = tensors_.size();
        tensors_.push_back(std::move(tt.t));
    }
}

const GTensor* GGUF::find(std::string_view name) const {
    auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &tensors_[it->second];
}
const GTensor& GGUF::get(std::string_view name) const {
    auto* t = find(name); if (!t) throw std::runtime_error("gguf: missing tensor " + std::string(name)); return *t;
}
const GValue* GGUF::kv(std::string_view key) const {
    auto it = kv_.find(std::string(key)); return it == kv_.end() ? nullptr : &it->second;
}
uint32_t GGUF::kv_u32(std::string_view key) const { auto* v = kv(key); if (!v) throw std::runtime_error("gguf: missing kv " + std::string(key)); return (uint32_t)(v->kind == GValue::I32 ? v->i : v->u); }
float GGUF::kv_f32(std::string_view key) const { auto* v = kv(key); if (!v) throw std::runtime_error("gguf: missing kv " + std::string(key)); return (float)v->f; }
std::string GGUF::kv_str(std::string_view key) const { auto* v = kv(key); if (!v) throw std::runtime_error("gguf: missing kv " + std::string(key)); return v->s; }
void GGUF::release(const GTensor& t) const {
    uintptr_t a = ((uintptr_t)t.data + 4095) & ~(uintptr_t)4095, e = ((uintptr_t)t.data + t.nbytes) & ~(uintptr_t)4095;
    if (e > a) madvise((void*)a, e - a, MADV_DONTNEED);                              // unmap from this process
    posix_fadvise(maps_[t.map_index].fd, (off_t)t.file_off, (off_t)t.nbytes, POSIX_FADV_DONTNEED);   // and drop the page cache
}
void GGUF::read_tensor(const GTensor& t, void* dst) const {
    const int fd = maps_[t.map_index].fd;
    size_t done = 0;
    while (done < t.nbytes) {
        const ssize_t r = pread(fd, (char*)dst + done, t.nbytes - done, (off_t)(t.file_off + done));
        if (r <= 0) throw std::runtime_error("gguf: pread failed " + t.name);
        done += (size_t)r;
    }
    posix_fadvise(fd, (off_t)t.file_off, (off_t)t.nbytes, POSIX_FADV_DONTNEED);
}
void GGUF::prefetch(const GTensor& t) const {
    uintptr_t a = (uintptr_t)t.data & ~(uintptr_t)4095;
    madvise((void*)a, t.nbytes + ((uintptr_t)t.data - a), MADV_WILLNEED);
}

}  // namespace hip
