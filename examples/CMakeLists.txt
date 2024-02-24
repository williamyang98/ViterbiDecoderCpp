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
    target_link_libraries(${target} PRIVATE getopt viterbi)
endfunction()

if(NOT WIN32)
    find_package(Threads REQUIRED)
    add_compile_options(-pthread)
    link_libraries(Threads::Threads)
endif()

create_example_target(run_tests)
create_example_target(run_benchmark)
create_example_target(run_simple)
create_example_target(run_punctured_decoder)
create_example_target(run_snr_ber)