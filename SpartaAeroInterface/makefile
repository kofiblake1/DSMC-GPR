
# Path to SPARTA library directory
SPARTA_LIB_DIR := $(HOME)/sparta-20Jan2025/src

CXX := mpic++
CXXFLAGS := -O3 -Wall -Wextra -Wconversion -std=c++17 -I$(SPARTA_LIB_DIR)
TARGET := main
OBJS := main.o SpartaAeroInterface.o sparta_utils.o
INCS := SpartaAeroInterface.hpp sparta_utils.hpp

# Linker flags: add rpath so the program can find libsparta_mpi.so at runtime
LDFLAGS := -Wl,-rpath=$(SPARTA_LIB_DIR) -L$(SPARTA_LIB_DIR) -lsparta_mpi

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp $(INCS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(TARGET)
