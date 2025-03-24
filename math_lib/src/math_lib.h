#ifndef MATH_LIB_H
#define MATH_LIB_H

// Include all headers in "math_lib" folder
#include "matrix.h"

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

Matrix normalize(const Matrix& m, const int axis);  // test

Matrix softmax(const Matrix& m, const int axis);  // test

Matrix positional_encoder(const Matrix input_embeddings, int d_model);  // test

Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V, bool masked);  // test

Matrix max(float n, const Matrix& m);  // test
Matrix max(const Matrix& m, float n);  // test
}


#endif // MATH_LIB_H