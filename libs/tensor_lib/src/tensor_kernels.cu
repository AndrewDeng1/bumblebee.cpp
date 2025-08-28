#include<iostream>
#include<vector>
#include "tensor.h"
using namespace std;

struct TensorStruct {
    int* shape;
    int shape_size;
    size_t shape_size_bytes;
    int* strides;
    int strides_size;
    size_t strides_size_bytes;
    float* data;
    int data_size;
    size_t data_size_bytes;
    
    TensorStruct(){}

    // Constructor that takes a tensor
    TensorStruct(shared_ptr<Tensor> t) {
        shape_size = t->shape.size();
        shape_size_bytes = shape_size*sizeof(int);
        shape = new int[shape_size];
        for(int i = 0; i < shape_size; i++) {
            shape[i] = t->shape[i];
        }
        
        strides_size = t->strides.size();
        strides_size_bytes = strides_size*sizeof(int);
        strides = new int[strides_size];
        for(int i = 0; i < strides_size; i++) {
            strides[i] = t->strides[i];
        }
        
        data = t->data;
        data_size = t->size();
        data_size_bytes = data_size*sizeof(float);
    }

    // Constructor that takes a tensor
    TensorStruct(float* t_data, int t_data_size, vector<int>& t_shape, vector<int>& t_strides) {
        shape_size = t_shape.size();
        shape = new int[shape_size];
        for(int i = 0; i < shape_size; i++) {
            shape[i] = t_shape[i];
        }
        
        strides_size = t_strides.size();
        strides = new int[strides_size];
        for(int i = 0; i < strides_size; i++) {
            strides[i] = t_strides[i];
        }
        
        data = t_data;
        data_size = t_data_size;
    }
    
    // Destructor to free memory
    ~TensorStruct() {
        delete[] shape;
        delete[] strides;
    }

    // Unnecessary as long as the shape and strides for the tensor are unaffected by operation,
    // assuming you're doing standard 1. Create TensorStruct 2. cudaMallocTensorStruct
    // 3. cudaMemcpyTensorStruct 4. Operation 5. cudaMemcpyTensorSTruct back 6. cudaFreeTensorStruct
    void toTensor(shared_ptr<Tensor> t){
        t->data = data;
        t->shape = vector<int>();
        for(int i=0; i<shape_size; i++){
            t->shape.push_back(shape[i]);
        }
        t->strides = vector<int>();
        for(int i=0; i<strides_size; i++){
            t->strides.push_back(strides[i]);
        }
    }
};

void cudaMallocTensorStruct(TensorStruct& a, TensorStruct& b){
    cudaMalloc(&a.shape, b.shape_size_bytes);
    cudaMalloc(&a.strides, b.strides_size_bytes);
    cudaMalloc(&a.data, b.data_size_bytes);
}

void cudaMemcpyTensorStruct(TensorStruct targ, TensorStruct src, cudaMemcpyKind dir){
    cudaMemcpy(targ.shape, src.shape, src.shape_size_bytes, dir);
    cudaMemcpy(targ.strides, src.strides, src.strides_size_bytes, dir);
    cudaMemcpy(targ.data, src.data, src.data_size_bytes, dir);
}

void cudaFreeTensorStruct(TensorStruct t){
    cudaFree(t.data);
    cudaFree(t.shape);
    cudaFree(t.strides);
}

__global__ void addKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]+b[idx];
    }
}

void launchAdd(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    addKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void subtractKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]-b[idx];
    }
}

void launchSubtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    subtractKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void multiplyKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]*b[idx];
    }
}

void launchMultiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    multiplyKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void divideKernel(float* a, float* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        c[idx]=a[idx]/b[idx];
    }
}

void launchDivide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result){
    float *d_a, *d_b, *d_c;
    int N = result->size();
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a->data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b->data, size, cudaMemcpyHostToDevice);

    divideKernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(result->data, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void broadcastKernel(TensorStruct a, TensorStruct b, bool matmul){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("idx: %d\n", idx);

    if(idx<b.data_size){
        int curr = idx;
        int cnt = 0;
        printf("(%d) start for. b.shape_size: %d, matmul: %d\n", idx, b.shape_size, matmul);
        for(int j=0; j<b.shape_size-2*matmul; j++) {
            printf("(%d) iter\n", idx);
            printf("(%d) b.strides: %d\n", idx, b.strides);
            printf("(%d) b.strides[j]: %d\n", idx, b.strides[j]);
            int dim = curr/b.strides[j];
            printf("(%d) dim: %d\n", idx, dim);
            curr %= b.strides[j];
            if(a.shape[j] == 1) {
                cnt += 0;  // Don't add to index for broadcasted dimensions
            } else {
                cnt += a.strides[j] * dim;
            }
        }
        printf("(%d) end for\n", idx);
        b.data[idx] = a.data[cnt];
    }
}

void launchBroadcast(shared_ptr<Tensor>a, shared_ptr<Tensor>b, vector<int>& padded_shape, vector<int>& padded_strides, bool matmul){
    TensorStruct a_struct{a->data, a->size(), padded_shape, padded_strides};
    TensorStruct b_struct{b};
    int N = b->size();
    printf("a");
    TensorStruct d_a_struct{};
    TensorStruct d_b_struct{};
    printf("b");
    
    cudaMallocTensorStruct(d_a_struct, a_struct);
    cudaMallocTensorStruct(d_b_struct, b_struct);
    printf("c");
    
    // Populate scalar fields in device-pointer structs (passed by value to kernel)
    d_a_struct.shape_size = a_struct.shape_size;
    d_a_struct.shape_size_bytes = a_struct.shape_size_bytes;
    d_a_struct.strides_size = a_struct.strides_size;
    d_a_struct.strides_size_bytes = a_struct.strides_size_bytes;
    d_a_struct.data_size = a_struct.data_size;
    d_a_struct.data_size_bytes = a_struct.data_size_bytes;

    d_b_struct.shape_size = b_struct.shape_size;
    d_b_struct.shape_size_bytes = b_struct.shape_size_bytes;
    d_b_struct.strides_size = b_struct.strides_size;
    d_b_struct.strides_size_bytes = b_struct.strides_size_bytes;
    d_b_struct.data_size = b_struct.data_size;
    d_b_struct.data_size_bytes = b_struct.data_size_bytes;
    printf("c0.5");
    // Copy input metadata and input data to device. For b, shape/strides are needed
    cudaMemcpyTensorStruct(d_a_struct, a_struct, cudaMemcpyHostToDevice);
    // cudaMemcpyTensorStruct(d_b_struct, b_struct, cudaMemcpyHostToDevice);
    printf("c1");
    broadcastKernel<<<(N+255)/256, 256>>>(d_a_struct, d_b_struct, matmul);
    printf("c2");
    cudaMemcpyTensorStruct(b_struct, d_b_struct, cudaMemcpyDeviceToHost);
    printf("d");
    
    // Unnecessary, since shape and strides isn't affected
    b_struct.toTensor(b);
    
    printf("e");
    cudaFreeTensorStruct(d_a_struct);
    cudaFreeTensorStruct(d_b_struct);
    printf("f");
}