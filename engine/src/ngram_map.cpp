#include "ngram_map.h"
#include <algorithm>

namespace hip {
NgramMap::NgramMap(int size_key, int size_value, int min_hits) : n_(size_key), m_(size_value), min_hits_(min_hits) { key_map_.assign(HASH_SIZE, 0); }

void NgramMap::begin(const std::vector<int>& tokens) {
    const size_t size_begin = tokens.size();
    size_t idx_begin_cleanup = size_last_begin_;
    if (idx_begin_cleanup > size_begin) idx_begin_cleanup = size_begin > (size_t)(n_ + m_) ? size_begin - n_ - m_ : 0;
    if (size_begin < idx_last_check_) {   // history shrank (new prompt): drop entries past the common prefix
        for (auto& e : key_map_) if (e != 0 && e >= idx_begin_cleanup) e = 0;
        key_map_last_idx_ = idx_begin_cleanup > 0 ? (uint32_t)(idx_begin_cleanup - 1) : 0;
        for (int i = (int)keys_.size() - 1; i >= 0; --i) {
            Key& key = keys_[i];
            if (key.key_idx >= idx_begin_cleanup) { keys_.erase(keys_.begin() + i); continue; }
            for (int j = MAX_VALUES - 1; j >= 0; --j) {
                Value& v = key.values[j];
                if (v.idx != 0 && v.idx >= idx_begin_cleanup) { for (int k = j; k < MAX_VALUES - 1; ++k) key.values[k] = key.values[k + 1]; key.values[MAX_VALUES - 1] = Value{}; }
            }
            if (key.values[0].idx == 0) keys_.erase(keys_.begin() + i);
        }
    }
    idx_last_check_ = size_begin; size_last_begin_ = size_begin;
}

std::vector<int> NgramMap::draft(const std::vector<int>& inp, int sampled) {
    std::vector<int> out;
    last_draft_created_ = false; last_key_ = 0; last_value_ = 0;
    const size_t cur_len = inp.size(); const size_t n = n_, m = m_;
    if (cur_len < 2 * n + m) return out;
    idx_last_check_ = cur_len;
    std::vector<int> key; key.reserve(n);
    for (size_t j = cur_len - n + 1; j < cur_len; ++j) key.push_back(inp[j]);
    key.push_back(sampled);
    size_t match_pos = 0;
    { const uint32_t idx_key = key_map_[hash(key, 0, n) % HASH_SIZE];
      if (idx_key != 0 && idx_key < cur_len - n - m - 1 && match_at(inp, idx_key, key)) match_pos = idx_key; }
    if (match_pos == 0 && size_last_begin_ > n + m + 1)
        for (size_t j = size_last_begin_ - n - m - 1; j > key_map_last_idx_; --j) if (match_at(inp, j, key)) { match_pos = j; break; }
    if (match_pos == 0)
        for (size_t j = cur_len - n - m - 1; j > size_last_begin_ && j > key_map_last_idx_; --j) if (match_at(inp, j, key)) { match_pos = j; break; }
    // index the n-grams not yet hashed (lowest index wins on collision)
    if (size_last_begin_ > n + m + 1)
        for (size_t j = size_last_begin_ - n - m - 1; j > key_map_last_idx_; --j) { uint32_t& e = key_map_[hash(inp, j, n) % HASH_SIZE]; if (e == 0) e = (uint32_t)j; }
    for (size_t j = cur_len - n - m - 1; j > size_last_begin_ && j > key_map_last_idx_; --j) { uint32_t& e = key_map_[hash(inp, j, n) % HASH_SIZE]; if (e == 0) e = (uint32_t)j; }
    key_map_last_idx_ = std::max((uint32_t)(cur_len - n - m - 1), key_map_last_idx_);
    if (match_pos == 0) return out;
    size_t ko = keys_.size();
    for (size_t i = 0; i < keys_.size(); ++i) if (match_at(inp, keys_[i].key_idx, key)) { ko = i; break; }
    if (ko == keys_.size()) { Key k{}; k.key_idx = match_pos; k.stat_idx = 0; k.key_num = 0; for (auto& v : k.values) { v.num = 0; v.n_accepted = (int16_t)m; } keys_.push_back(k); }
    Key& cur = keys_[ko];
    cur.key_num = (uint16_t)std::min((int)cur.key_num + 1, MAX_COUNT);
    if (cur.key_num < min_hits_) return out;
    for (size_t i = cur.stat_idx; i <= match_pos; ++i) {   // value statistics over every earlier occurrence of the key
        if (!match_at(inp, i, key)) continue;
        const size_t vb = i + n; int iv = -1;
        for (int v = 0; v < MAX_VALUES; ++v) {
            Value& val = cur.values[v];
            if (val.idx == 0) { val.idx = vb; val.num = 0; val.n_accepted = (int16_t)m; iv = v; break; }
            bool same = true; for (size_t j = 0; j < m; ++j) if (inp[vb + j] != inp[val.idx + j]) { same = false; break; }
            if (same) { iv = v; break; }
        }
        if (iv >= 0) cur.values[iv].num = (uint16_t)std::min((int)cur.values[iv].num + 1, MAX_COUNT);
    }
    cur.stat_idx = match_pos;
    uint16_t max_occur = 0; int slot_max = 0;
    for (int v = 0; v < MAX_VALUES; ++v) if (cur.values[v].num > max_occur) { max_occur = cur.values[v].num; slot_max = v; }
    uint32_t sum_occur = 0; for (int v = 0; v < MAX_VALUES; ++v) if (v != slot_max) sum_occur += cur.values[v].num;
    if (sum_occur > 0 && max_occur < 2 * sum_occur) return out;
    const int nd = std::min((int)m, (int)cur.values[slot_max].n_accepted);
    for (int i = 0; i < nd; ++i) out.push_back(inp[match_pos + n + i]);
    last_draft_created_ = true; last_key_ = ko; last_value_ = slot_max;
    return out;
}

void NgramMap::accept(int n_accepted) {
    if (!last_draft_created_) return;
    keys_[last_key_].values[last_value_].n_accepted = (int16_t)n_accepted;
}
}  // namespace hip
