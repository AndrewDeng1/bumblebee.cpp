#include "math_lib.h"
#include <cmath> // For std::exp

using namespace std;

namespace math_lib {

// Sigmoid function implementation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Optional: Sigmoid function for vectors (element-wise application)
Matrix sigmoid(const Matrix& m) {
    Matrix ret = Matrix(m.getRows(), m.getCols());
    for (size_t i = 0; i < m.getRows(); i++) {
        for (size_t j = 0; j < m.getCols(); j++) {
            ret[i][j] = sigmoid(m[i][j]);
        }
    }
    return ret;
}

float average_error(const Matrix& y_true, const Matrix& y_pred) {

    assert(y_true.getRows() == y_pred.getRows()&&y_true.getCols()==1&&y_pred.getCols()==1&&"y_true and y_pred must both be vector shape (nx1) and have equal number of rows.");

    float error = 0;
    for (size_t i = 0; i < y_true.getRows(); i++) {
        error += abs(y_true[i][0] - y_pred[i][0]);
    }
    return error / y_true.getRows();
}

float squared_euclidean_distance(const vector<float>& v1, const vector<float> v2){
    return squared_euclidean_distance(Matrix(v1), Matrix(v2));
}

float squared_euclidean_distance(const Matrix& v1, const Matrix& v2){
    assert(v1.getCols() == 1 && v2.getCols() == 1 && v1.getRows() == v2.getRows() && "Both vectors must be column vectors of the same size.");

    float sm=0;
    for(size_t i=0; i<v1.getRows(); i++){
        sm += pow(v1[i][0]-v2[i][0], 2);
    }

    return sm;
}

}
