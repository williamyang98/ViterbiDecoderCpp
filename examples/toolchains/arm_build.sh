#!/bin/sh
cmake . -B build-arm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=./toolchains/aarch64-linux-gnu.toolchain.cmake