# libs/math_lib/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Set source files (without data_frame)
set(MATH_LIB_SOURCES
    src/math_lib.cpp
    src/matrix.cpp
)

# Create library
add_library(math_lib ${MATH_LIB_SOURCES})

# Set include directories
target_include_directories(math_lib
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# MSVC specific settings
if(MSVC)
    # Keep warnings but disable specific conversion warnings
    target_compile_options(math_lib PRIVATE 
        /W4
        /wd4244  # double to float conversion
        /wd4267  # size_t conversion
    )
endif()

# Install rules (if needed)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS math_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)