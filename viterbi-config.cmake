cmake_minimum_required(VERSION 3.10)
project(viterbi)

add_library(viterbi INTERFACE)
target_include_directories(viterbi INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)