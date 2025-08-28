cmake_minimum_required(VERSION 3.10)

# Optional CUDA support for tensor kernels
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

# Source files for tensor library
set(TENSOR_LIB_SOURCES
    src/tensor.cpp
)

# If CUDA is available and the kernels exist, add them
set(TENSOR_KERNELS_CU ${CMAKE_SOURCE_DIR}/tests/unit/tensor_lib/tensor_kernels.cu)
if(CMAKE_CUDA_COMPILER AND EXISTS ${TENSOR_KERNELS_CU})
  list(APPEND TENSOR_LIB_SOURCES ${TENSOR_KERNELS_CU})
endif()

# Create library
add_library(tensor_lib ${TENSOR_LIB_SOURCES})

# Include directories
target_include_directories(tensor_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        # For tensor_kernels.h used by src/tensor.cpp
        ${CMAKE_SOURCE_DIR}/tests/unit/tensor_lib
)

# MSVC specific settings
if(MSVC)
    target_compile_options(tensor_lib PRIVATE 
        /W4
        /wd4244  # double to float conversion
        /wd4267  # size_t conversion
    )
endif()

# Install rules (if needed)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS tensor_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)