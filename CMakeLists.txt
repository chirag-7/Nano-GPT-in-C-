cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0074 NEW)
project(nanogpt)

# Diagnostic Messages to Verify Configuration
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "APPLE: ${APPLE}")
message(STATUS "Initial CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")

# Set C++17 Standard Globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find LibTorch
find_package(Torch REQUIRED)

# Diagnostic Messages to Verify Torch Configuration
message(STATUS "TORCH_LIBRARIES before cleaning: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")

# Append LibTorch's C++ Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS after appending TORCH_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# Define the Executable and Include Only Source Files
add_executable(nanogpt main.cpp dataset.cpp)

# Link LibTorch libraries using the imported target Torch::Torch
target_link_libraries(nanogpt "${TORCH_LIBRARIES}")

# Alternatively, use the imported target directly
# target_link_libraries(nanogpt Torch::Torch)

# Enable Compiler Warnings and Treat Them as Errors
target_compile_options(nanogpt PRIVATE -Wall -Wextra -Werror)

# Optional: Specify Include Directories if Headers are in a Separate Directory
# target_include_directories(nanogpt PRIVATE include)
