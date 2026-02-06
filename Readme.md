# Slow Feature Analysis

This is an implementation of slow feature analysis, based on James Stone and Alistair Bray's 1995 paper: "A learning rule for extracting spatio-temporal invariances"

## Building the project

The project uses CMake for building.

```bash
mkdir build
cd build
cmake ..
make
```

This will generate two executables:
- `main`: The main simulation program.
- `run_tests`: The test suite.

## Repository Structure

- `src/`: Source files.
- `include/`: Header files.
- `tests/`: Test files.
- `data/`: Output CSV files and other data.
- `CMakeLists.txt`: CMake configuration file.