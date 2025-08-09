nvcc -c tensor_kernels.cu -o tensor_kernels.o
nvcc tensor.cpp main.cpp tensor_kernels.o -o tensor_test
tensor_test.exe