#ifndef MATH_LIB_H
#define MATH_LIB_H

// Include all headers in "math_lib" folder
#include <math_lib/matrix.h>
#include <math_lib/tensor.h>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>
#include <string>

namespace math_lib {

// Sigmoid function prototype
float sigmoid(float x);

// Optional: Sigmoid function for vectors (element-wise application)
Matrix sigmoid(const Matrix& m);

// Average accuracy of predictions vs ground truth
float average_error(const Matrix& y_true, const Matrix& y_pred);

// Calculates squared euclidean distance between two vectors (or L_2^2 norm)
float squared_euclidean_distance(const vector<float>& v1, const vector<float> v2);
float squared_euclidean_distance(const Matrix& v1, const Matrix& v2);

float mean(const vector<float>& v);

float std_dev(const vector<float>& v);

shared_ptr<Tensor> add(const shared_ptr<Tensor>& t1, const shared_ptr<Tensor>& t2);

shared_ptr<Tensor> matmul(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);

shared_ptr<Tensor> normalize(const shared_ptr<Tensor>& x, float epsilon=1e-5);

shared_ptr<Tensor> relu(const shared_ptr<Tensor>& x);

shared_ptr<Tensor> softmax(const shared_ptr<Tensor>& x, float epsilon=1e-5);

Matrix positional_encoder(const Matrix input_embeddings, int d_model);

shared_ptr<Tensor> add_and_norm(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B);

shared_ptr<Tensor> concat(const vector<shared_ptr<Tensor>>& tensors, int axis);

shared_ptr<Tensor> attention(
    const shared_ptr<Tensor>& Q,
    const shared_ptr<Tensor>& K,
    const shared_ptr<Tensor>& V,
    int d_k,
    bool masked = false
);

Matrix max(float n, const Matrix& m);
Matrix max(const Matrix& m, float n);

void xavier_uniform_initialization(Matrix& m, int d_in, int d_out);

// Embed a sequence of tokens using a token-to-embedding map
shared_ptr<Tensor> embed(
    const vector<string>& input_sequence,
    const unordered_map<string, vector<float>>& token_embeddings,
    int d_model
);
}


#endif // MATH_LIB_H