// Minimal GGUF v3 reader: mmap the file(s), parse the KV header and tensor table, expose raw
// pointers. No dequantisation, no ops. Sharded files are joined by tensor name.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hip {

enum class GType : uint32_t {  // ggml_type numbering (subset we ever see)
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22,
    IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, BF16 = 30,
};
const char* gtype_name(GType t);
// (block size in elements, bytes per block)
std::pair<uint32_t, uint32_t> gtype_block(GType t);
inline size_t gtype_row_bytes(GType t, uint64_t ne0) {
    auto [bs, tb] = gtype_block(t);
    return (size_t)(ne0 / bs) * tb;
}

struct GTensor {
    std::string name;
    GType type;
    uint32_t n_dims;
    uint64_t ne[4] = {1, 1, 1, 1};   // ne[0] is the contiguous (K) dimension
    size_t nbytes;
    const uint8_t* data;             // into the mmap
    int map_index = 0;               // which shard
    uint64_t file_off = 0;           // offset of the data in that shard
};

struct GValue {  // scalar / string / array-of-scalar KV
    enum Kind { U32, I32, F32, BOOL, STR, U64, I64, F64, ARR } kind;
    uint64_t u = 0; int64_t i = 0; double f = 0; std::string s;
    std::vector<GValue> arr;
};

class GGUF {
public:
    // Opens one file, or every shard of a split (pass any shard).
    explicit GGUF(const std::string& path);
    ~GGUF();
    GGUF(const GGUF&) = delete; GGUF& operator=(const GGUF&) = delete;

    const GTensor* find(std::string_view name) const;
    const GTensor& get(std::string_view name) const;  // throws if missing
    const std::vector<GTensor>& tensors() const { return tensors_; }
    bool has(const std::string& name) const { return index_.count(name) != 0; }
    const GValue* kv(std::string_view key) const;
    uint32_t kv_u32(std::string_view key) const;
    float kv_f32(std::string_view key) const;
    std::string kv_str(std::string_view key) const;
    std::string arch() const { return kv_str("general.architecture"); }
    size_t mapped_bytes() const { return mapped_bytes_; }
    // Advise the kernel that these tensors will be streamed sequentially (readahead).
    void prefetch(const GTensor& t) const;
    // Drop the page cache for a tensor already copied to the GPU (frees host RAM; the mmap stays valid).
    void release(const GTensor& t) const;
    // Read a tensor's bytes with pread into a host buffer (no page-cache growth: the range is dropped right after).
    // Use this for weights that are uploaded to the GPU; keep the mmap only for on-demand row gathers.
    void read_tensor(const GTensor& t, void* dst) const;

private:
    void open_one(const std::string& path);
    struct Map { void* p; size_t len; int fd; };
    std::vector<Map> maps_;
    std::vector<GTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::unordered_map<std::string, GValue> kv_;
    size_t mapped_bytes_ = 0;
};

}  // namespace hip
