add_library(tokenizer_lib
    src/tokenizer.cpp
)

target_include_directories(tokenizer_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
) 