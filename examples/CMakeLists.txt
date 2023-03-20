cmake_minimum_required(VERSION 3.10)
project(viterbi_decoder)

set(viterbi_DIR ${CMAKE_CURRENT_LIST_DIR}/../)
find_package(viterbi CONFIG REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

add_library(getopt STATIC ${SRC_DIR}/getopt/getopt.c)
target_include_directories(getopt PRIVATE ${SRC_DIR}/getopt)
target_compile_features(getopt PRIVATE cxx_std_11)

function(create_example_target target)
    add_executable(${target} ${SRC_DIR}/${target}.cpp)
    target_include_directories(${target} PRIVATE ${SRC_DIR})
    target_compile_features(${target} PRIVATE cxx_std_17)
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        # x86 processors 
        target_compile_definitions(${target} PRIVATE VITERBI_SIMD_X86)
        if(MSVC)
            target_compile_options(${target} PRIVATE /fp:fast /arch:AVX2 /W3)
            target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        else()
            target_compile_options(${target} PRIVATE -mavx2 -mfma -ffast-math)
        endif()
    else()
        # arm processors
        target_compile_definitions(${target} PRIVATE VITERBI_SIMD_ARM)
        target_compile_options(${target} PRIVATE -ffast-math)
    endif()
    target_link_libraries(${target} PRIVATE getopt viterbi)
endfunction()

create_example_target(run_simple)
create_example_target(run_decoder)
create_example_target(run_benchmark)
create_example_target(run_punctured_decoder)
create_example_target(run_tests)