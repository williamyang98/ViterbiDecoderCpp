#pragma once
#include <stddef.h>

// NOTE: A limited std::span approximation
//       You should use a proper std::span implementation for your C++ standard version
template <typename T>
class span_t 
{
private:
    T* buf;
    size_t N;
public:
    span_t(T* _buf, size_t _N)
    : buf(_buf), N(_N)  {
        assert(buf != NULL);
    }

    template <typename U>
    span_t(U&& _buf)
    : buf(_buf.data()), N(_buf.size()) 
    {
        assert(buf != NULL);
    }

    T& operator[](const size_t i) const {
        return buf[i];
    }

    span_t<T> front(const size_t M) const {
        assert(buf != NULL);
        assert(N >= M);
        const size_t N_remain = N-M;
        return { &buf[M], N_remain };
    }
    span_t<T> back(const size_t M) const {
        assert(buf != NULL);
        assert(N >= M);
        const size_t N_start = N-M;
        return { &buf[N_start], M };
    }

    T* data() const { return buf; }
    T* begin() const { return buf; }
    T* end() const { return &buf[N]; }
    size_t size() const { return N; }
};
