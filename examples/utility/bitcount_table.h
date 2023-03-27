#pragma once

#include <stdint.h>

// Return number of 1s bits
class BitcountTable 
{
private:
    uint8_t* table;
    BitcountTable() {
        constexpr size_t N = 256;
        table = new uint8_t[N];
        for (size_t i = 0u; i < 256u; i++) {
            uint8_t x = (uint8_t)i;
            uint8_t count = 0;
            for (int j = 0u; j < 8; j++) {
                count += (x & 0b1);
                x = x >> 1;
            }
            table[i] = count;
        }
    }
    ~BitcountTable() {
        delete [] table;
    }
    BitcountTable(const BitcountTable&) = delete;
    BitcountTable(BitcountTable&&) = delete;
    BitcountTable& operator=(const BitcountTable&) = delete;
    BitcountTable& operator=(BitcountTable&&) = delete;
public:
    static 
    BitcountTable& get() {
        static auto bitcount_table = BitcountTable();
        return bitcount_table;
    }

    uint8_t parse(uint8_t x) {
        return table[x];
    }

    template <typename T>
    uint8_t parse(const T x) {
        static_assert(std::is_trivially_copyable<T>::value, "Not a copyable type" );
        constexpr size_t N = sizeof(x);
        auto* arr = reinterpret_cast<const uint8_t*>(&x);
        uint8_t count = 0u;
        for (size_t i = 0u; i < N; i++) {
            count += table[arr[i]];
        }
        return count;
    }
};