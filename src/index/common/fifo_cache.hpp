#ifndef IGNIS_FIFO_CACHE_HPP
#define IGNIS_FIFO_CACHE_HPP

#include <unordered_map>
#include <stdexcept>
#include <boost/circular_buffer.hpp>

/**
 * @brief A simple First-In, First-Out (FIFO) cache.
 *
 * This cache evicts the oldest inserted item when it reaches its maximum size.
 *
 * @tparam key_t The type of the keys.
 * @tparam value_t The type of the values.
 */
template<typename key_t, typename value_t>
class fifo_cache {
public:
    /**
     * @brief Constructs a fifo_cache with a specified maximum size.
     * @param max_size The maximum number of items the cache can hold.
     */
    fifo_cache(size_t max_size) :
        _keys(max_size) {
    }

    /**
     * @brief Inserts a key-value pair into the cache.
     *
     * If the key already exists, the operation is a no-op. If the cache is full,
     * the oldest item is evicted.
     *
     * @param key The key to insert.
     * @param value The value to associate with the key.
     */
    void put(const key_t& key, const value_t& value) {
        if (_cache_items_map.find(key) == _cache_items_map.end()) {
            if (_keys.full()) {
                _cache_items_map.erase(_keys.front());
            }
            _keys.push_back(key);
            _cache_items_map[key] = value;
        }
    }

    /**
     * @brief Retrieves the value associated with a given key.
     * @param key The key to look up.
     * @return A const reference to the value.
     * @throws std::range_error if the key is not found in the cache.
     */
    const value_t& get(const key_t& key) {
        auto it = _cache_items_map.find(key);
        if (it == _cache_items_map.end()) {
            throw std::range_error("There is no such key in cache");
        }
        return it->second;
    }

    /**
     * @brief Checks if a key exists in the cache.
     * @param key The key to check.
     * @return True if the key exists, false otherwise.
     */
    bool exists(const key_t& key) const {
        return _cache_items_map.find(key) != _cache_items_map.end();
    }

    /**
     * @brief Returns the current number of items in the cache.
     * @return The size of the cache.
     */
    size_t size() const {
        return _cache_items_map.size();
    }

private:
    std::unordered_map<key_t, value_t> _cache_items_map;
    boost::circular_buffer<key_t> _keys;
};

#endif // IGNIS_FIFO_CACHE_HPP
