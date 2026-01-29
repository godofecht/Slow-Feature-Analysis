cmake_minimum_required(VERSION 3.10)
project(SlowFeatureAnalysis)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Shared source files
set(SOURCES
    src/NN.cpp
    src/Network.cpp
    src/model.cpp
    src/sfa.cpp
)

# Main executable
add_executable(main src/main.cpp ${SOURCES})
target_link_libraries(main m)

# Test executable
add_executable(run_tests tests/tests.cpp ${SOURCES})
target_link_libraries(run_tests m)
