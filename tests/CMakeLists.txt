# Build individual test executables
file(GLOB TEST_SOURCES 
    "unit/math_lib/*.cpp"
    "unit/transformer_lib/*.cpp"
)

foreach(test_src ${TEST_SOURCES})
    # Get test name from filename (test_matrix.cpp -> test_matrix)
    get_filename_component(test_name ${test_src} NAME_WE)
    
    # Create executable
    add_executable(${test_name} ${test_src})
    
    # Link only project libraries
    target_link_libraries(${test_name} 
        PRIVATE 
            math_lib 
            transformer_lib
    )
    
    # Install to bin/ (optional)
    install(TARGETS ${test_name} DESTINATION bin)
endforeach()

# Optional: Add all tests to CTest (basic version)
enable_testing()
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()