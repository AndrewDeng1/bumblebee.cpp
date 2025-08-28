#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Tensor : public enable_shared_from_this<Tensor> {
    
    public:
        Tensor();
        Tensor(const vector<int>& shape, bool requires_grad=false);
        Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad=false);
        Tensor(const vector<int>& shape, float* data, bool requires_grad=false);
        Tensor(const vector<int>& shape, float num, bool requires_grad=false);
        Tensor(shared_ptr<Tensor> other);

        // // Member functions
        int size();
        // void backward();
        float& at(vector<int> indices);
        float& at(int index);
        vector<int> compute_strides(const vector<int>& shape);
        shared_ptr<Tensor> reshape(const vector<int>& new_shape);
        shared_ptr<Tensor> reduce_to_shape(const vector<int>& target_shape);
        shared_ptr<Tensor> sum(int axis, bool keepdims=true);
        shared_ptr<Tensor> transpose(int dim1, int dim2);
        shared_ptr<Tensor> mean(int axis, bool keepdims=true);
        shared_ptr<Tensor> variance_squared(int axis, bool keepdims=true);
        shared_ptr<Tensor> norm(int axis, bool keepdims=true);
        shared_ptr<Tensor> pow(float exponent);
        shared_ptr<Tensor> softmax(int axis);
        // shared_ptr<Tensor> log_softmax(int axis, bool keepdims=true);
        // shared_ptr<Tensor> negative_log_likelihood(const shared_ptr<Tensor>& y_true);
        shared_ptr<Tensor> cross_entropy(const shared_ptr<Tensor>& y_true, int axis, bool keepdims=true);
        void xavier_uniform_initialization(shared_ptr<Tensor>& t);
        void zero_grad();
        void step(float learning_rate);
        void backward();
        void print();

        // Broadcasting and reduction operations
        shared_ptr<Tensor> broadcast(const vector<int>& new_shape, bool matmul = false);

        // // Member variables
        bool requires_grad;
        vector<int> shape;
        vector<int> strides;
        float* data;
        shared_ptr<Tensor> grad;
        vector<shared_ptr<Tensor>> parents;
        function<void()> backward_fn;
};

// // Free function declarations
shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator/(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator+=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator-=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator*=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor>& operator/=(shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);

// Scalar-tensor operators
shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator+(float A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator-(float A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& A);
shared_ptr<Tensor> operator*(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator*(float A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> operator/(const shared_ptr<Tensor>& A, float B);
shared_ptr<Tensor> operator/(float A, const shared_ptr<Tensor>& B);

shared_ptr<Tensor> matmul(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);

shared_ptr<Tensor> relu(const shared_ptr<Tensor>& A);
shared_ptr<Tensor> sigmoid(const shared_ptr<Tensor>& A);
shared_ptr<Tensor> tanh(const shared_ptr<Tensor>& A);

shared_ptr<Tensor> add_and_norm(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);
shared_ptr<Tensor> attention(const shared_ptr<Tensor>& Q, const shared_ptr<Tensor>& K, const shared_ptr<Tensor>& V, int d_k, bool masked);

// Global functions
bool is_broadcastable(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul = false);
vector<int> get_broadcast_shape(const vector<int>& A_shape, const vector<int>& B_shape, bool matmul = false);

shared_ptr<Tensor> positional_encoder(const shared_ptr<Tensor>& input_embeddings, int d_model);

#endif // TENSOR_H