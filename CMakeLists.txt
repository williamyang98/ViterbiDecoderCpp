cmake_minimum_required(VERSION 3.10)
project(viterbi_decoder)

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /arch:AVX2 /W3")
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -ffast-math -fpermissive")
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(run_decoder
    ${SRC_DIR}/examples/run_decoder.cpp  
    ${SRC_DIR}/examples/getopt/getopt.c
)
target_include_directories(run_decoder PUBLIC ${SRC_DIR} ${SRC_DIR}/examples)
set_target_properties(run_decoder PROPERTIES CXX_STANDARD 17)

add_executable(run_benchmark
    ${SRC_DIR}/examples/run_benchmark.cpp  
    ${SRC_DIR}/examples/getopt/getopt.c
)
target_include_directories(run_benchmark PUBLIC ${SRC_DIR} ${SRC_DIR}/examples)
set_target_properties(run_benchmark PROPERTIES CXX_STANDARD 17)

add_executable(run_punctured_decoder
    ${SRC_DIR}/examples/run_punctured_decoder.cpp  
    ${SRC_DIR}/examples/getopt/getopt.c
)
target_include_directories(run_punctured_decoder PUBLIC ${SRC_DIR} ${SRC_DIR}/examples)
set_target_properties(run_punctured_decoder PROPERTIES CXX_STANDARD 17)