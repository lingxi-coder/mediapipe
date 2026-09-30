// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef MEDIAPIPE_UTIL_TILING_CACHE_UTILS_H_
#define MEDIAPIPE_UTIL_TILING_CACHE_UTILS_H_

#include <cstdint>
#include <cstring>
#include <list>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/cache_stats.h"

namespace mediapipe {

// A stable, collision-safe cache key. `bytes` is a type-tagged, length-delimited
// encoding of all key fields; equality/lookup compares the FULL bytes (never a
// truncated hash), so two different inputs cannot collide to the same entry.
// `fingerprint` is a fast 64-bit digest kept for diagnostics only.
struct StableCacheKey {
  std::string bytes;
  uint64_t fingerprint = 0;
  bool operator==(const StableCacheKey& other) const {
    return bytes == other.bytes;
  }
};

// Folds key fields into a StableCacheKey. Floats are bit-cast to their integer
// representation so identical values encode identically across calls and runs
// (no float formatting drift). Each field is type-tagged so e.g. AddInt(1) and
// AddString("\x01") do not encode to the same bytes.
class StableKeyBuilder {
 public:
  StableKeyBuilder& AddInt(int64_t value) {
    buf_.push_back('\x01');
    AppendRaw(&value, sizeof(value));
    return *this;
  }
  StableKeyBuilder& AddBool(bool value) {
    buf_.push_back('\x02');
    const uint8_t b = value ? 1 : 0;
    AppendRaw(&b, sizeof(b));
    return *this;
  }
  StableKeyBuilder& AddFloat(float value) {
    buf_.push_back('\x03');
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));  // bit-cast, stable encoding
    AppendRaw(&bits, sizeof(bits));
    return *this;
  }
  StableKeyBuilder& AddString(absl::string_view value) {
    buf_.push_back('\x04');
    const int64_t n = static_cast<int64_t>(value.size());
    AppendRaw(&n, sizeof(n));  // length-delimited: avoids boundary ambiguity
    buf_.append(value.data(), value.size());
    return *this;
  }
  StableCacheKey Build() const {
    StableCacheKey key;
    key.bytes = buf_;
    key.fingerprint = Fnv1a(buf_);
    return key;
  }

 private:
  void AppendRaw(const void* p, size_t n) {
    buf_.append(reinterpret_cast<const char*>(p), n);
  }
  static uint64_t Fnv1a(const std::string& s) {
    uint64_t h = 14695981039346656037ull;  // FNV-1a 64-bit offset basis
    for (unsigned char c : s) {
      h ^= c;
      h *= 1099511628211ull;  // FNV-1a 64-bit prime
    }
    return h;
  }
  std::string buf_;
};

// Bounded LRU memo. capacity == 0 ⇒ disabled: Get always misses, Put is a no-op,
// stats stay zero — so a caller behaves exactly as if there were no cache.
// Calculator-local and single-threaded (one Process at a time); no locking.
// The map is keyed on the full key bytes, so a fingerprint collision cannot
// return the wrong value.
template <typename Value>
class BoundedLruCache {
 public:
  explicit BoundedLruCache(size_t capacity) : capacity_(capacity) {}

  bool enabled() const { return capacity_ > 0; }

  // Returns nullptr on miss. The returned pointer is valid only until the next
  // Put() call (a Put may evict this entry); do not hold it across Put().
  const Value* Get(const StableCacheKey& key) {
    if (capacity_ == 0) return nullptr;
    auto it = index_.find(key.bytes);
    if (it == index_.end()) {
      ++stats_.misses;
      return nullptr;
    }
    order_.splice(order_.begin(), order_, it->second);  // promote to MRU
    ++stats_.hits;
    return &it->second->second;
  }

  void Put(const StableCacheKey& key, Value value) {
    if (capacity_ == 0) return;
    auto it = index_.find(key.bytes);
    if (it != index_.end()) {
      it->second->second = std::move(value);
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    order_.emplace_front(key.bytes, std::move(value));
    index_[key.bytes] = order_.begin();
    ++stats_.inserts;
    if (index_.size() > capacity_) {
      const std::string& lru_key = order_.back().first;
      index_.erase(lru_key);
      order_.pop_back();
      ++stats_.evictions;
    }
  }

  const CacheStats& stats() const { return stats_; }

 private:
  using Entry = std::pair<std::string, Value>;  // (key bytes, value)
  size_t capacity_;
  std::list<Entry> order_;  // front = most-recently-used
  absl::flat_hash_map<std::string, typename std::list<Entry>::iterator> index_;
  CacheStats stats_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_TILING_CACHE_UTILS_H_
