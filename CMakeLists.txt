cmake_minimum_required(VERSION 3.10)
project(SlowFeatureAnalysis)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -ftree-vectorize -fopenmp")
find_package(OpenMP REQUIRED)

include_directories(include)

# Shared source files
set(SOURCES
    src/Network.cpp
    src/model.cpp
    src/sfa.cpp
)

# Main executable
add_executable(main src/main.cpp ${SOURCES})
target_link_libraries(main m OpenMP::OpenMP_CXX)

# Test executable
add_executable(run_tests tests/tests.cpp ${SOURCES})
target_link_libraries(run_tests m OpenMP::OpenMP_CXX)
