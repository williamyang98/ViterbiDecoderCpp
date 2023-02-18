#pragma once

#include <stdint.h>
#include <stddef.h>

class ConvolutionalEncoder 
{
public:
    const size_t K;
    const size_t R;
public:
    ConvolutionalEncoder(const size_t constraint_length, const size_t code_rate)
    : K(constraint_length), R(code_rate)
    {

    }

    virtual void reset() = 0;

    // Output R bytes for each input byte
    virtual void consume_byte(const uint8_t x, uint8_t* y) = 0;
};

