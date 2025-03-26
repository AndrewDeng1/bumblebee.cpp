# libs/transformer_lib/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Source files (update with all your .cpp files)
set(TRANSFORMER_LIB_SOURCES
    src/decoder.cpp
    src/decoder_layer.cpp
    src/encoder.cpp
    src/encoder_layer.cpp
    src/feed_forward.cpp
    src/head.cpp
    src/linear.cpp
    src/multi_head_attention.cpp
    src/transformer.cpp
)

# Create the library
add_library(transformer_lib ${TRANSFORMER_LIB_SOURCES})

# Include directories
target_include_directories(transformer_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # Public API headers
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src  # For private includes
        ${CMAKE_SOURCE_DIR}/libs/math_lib/include  # Math lib headers
)

# Dependencies
target_link_libraries(transformer_lib
    PUBLIC
        math_lib  # Links and inherits math_lib's public includes
)

# MSVC-specific settings
if(MSVC)
    target_compile_options(transformer_lib PRIVATE /W4 /WX)
    target_compile_definitions(transformer_lib PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Install rules (optional)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS transformer_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)