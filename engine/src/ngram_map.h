// n-gram map speculation (port of llama.cpp common/ngram-map.cpp, "ngram-map-k4v"): the sequence's own history is the
// draft model. Key = the last size_key tokens (incl. the sampled one); if that n-gram occurred before, the m-gram that
// followed it is the draft, provided the key's dominant continuation is at least 2x all others (k4v statistics, up to
// 4 values per key) and the key has >= min_hits occurrences. accept() feeds back how much of the draft was accepted
// (later drafts from that value are shortened to it). Token ids only; the engine never sees it.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hip {
class NgramMap {
public:
    explicit NgramMap(int size_key = 12, int size_value = 48, int min_hits = 1);
    void begin(const std::vector<int>& tokens);   // start of a generation over this history (prompt)
    // history = prompt + generated tokens so far (without `sampled`); returns the draft (possibly empty)
    std::vector<int> draft(const std::vector<int>& history, int sampled);
    void accept(int n_accepted);
    int size_key() const { return n_; }
    int size_value() const { return m_; }
private:
    static constexpr int MAX_VALUES = 4, HASH_SIZE = 262144, MAX_COUNT = 16380;
    struct Value { size_t idx = 0; uint16_t num = 0; int16_t n_accepted = -1; };
    struct Key { size_t key_idx, stat_idx; uint16_t key_num; Value values[MAX_VALUES]; };
    static uint32_t hash(const std::vector<int>& t, size_t start, size_t len) { uint32_t h = 0; for (size_t i = 0; i < len; ++i) h = h * 2654435761u + (uint32_t)t[start + i]; return h; }
    bool match_at(const std::vector<int>& inp, size_t j, const std::vector<int>& key) const { for (size_t k = 0; k < key.size(); ++k) if (inp[j + k] != key[k]) return false; return true; }
    int n_, m_, min_hits_;
    std::vector<Key> keys_;
    std::vector<uint32_t> key_map_; uint32_t key_map_last_idx_ = 0;
    size_t size_last_begin_ = 0, idx_last_check_ = 0;
    bool last_draft_created_ = false; size_t last_key_ = 0; int last_value_ = 0;
};
}  // namespace hip
