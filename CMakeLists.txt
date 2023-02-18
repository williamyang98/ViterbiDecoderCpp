cmake_minimum_required(VERSION 3.10)
project(viterbi_decoder)

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /arch:AVX2 /W3")
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -ffast-math -Wall")
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(run_soft_decoding
    ${SRC_DIR}/examples/run_soft_decoding.cpp  
    ${SRC_DIR}/examples/getopt/getopt.c
)
target_include_directories(run_soft_decoding PUBLIC ${SRC_DIR} ${SRC_DIR}/examples)
set_target_properties(run_soft_decoding PROPERTIES CXX_STANDARD 17)

add_executable(run_hard_decoding
    ${SRC_DIR}/examples/run_hard_decoding.cpp  
    ${SRC_DIR}/examples/getopt/getopt.c
)
target_include_directories(run_hard_decoding PUBLIC ${SRC_DIR} ${SRC_DIR}/examples)
set_target_properties(run_hard_decoding PROPERTIES CXX_STANDARD 17)

add_executable(run_benchmark
    ${SRC_DIR}/examples/run_benchmark.cpp  
    ${SRC_DIR}/examples/getopt/getopt.c
)
target_include_directories(run_benchmark PUBLIC ${SRC_DIR} ${SRC_DIR}/examples)
set_target_properties(run_benchmark PROPERTIES CXX_STANDARD 17)