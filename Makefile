CXX = g++
CXXFLAGS = -std=c++11 -Wall -Iinclude
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%.o, $(SRCS))
TARGET = main
TEST_TARGET = run_tests

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(filter-out $(BIN_DIR)/tests.o, $(OBJS)) -o $@ $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(filter-out $(BIN_DIR)/main.o, $(OBJS)) $(BIN_DIR)/tests.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/tests.o: tests/tests.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR) $(TARGET)

.PHONY: all clean
