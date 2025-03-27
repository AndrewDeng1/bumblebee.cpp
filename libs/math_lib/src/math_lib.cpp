#include <math_lib/math_lib.h>

using namespace std;

namespace math_lib {

// Sigmoid function implementation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Optional: Sigmoid function for vectors (element-wise application)
Matrix sigmoid(const Matrix& m) {
    Matrix ret = Matrix(m.numRows(), m.numCols());
    for (size_t i = 0; i < m.numRows(); i++) {
        for (size_t j = 0; j < m.numCols(); j++) {
            ret[i][j] = sigmoid(m[i][j]);
        }
    }
    return ret;
}

float average_error(const Matrix& y_true, const Matrix& y_pred) {

    assert(y_true.numRows() == y_pred.numRows()&&y_true.numCols()==1&&y_pred.numCols()==1&&"y_true and y_pred must both be vector shape (nx1) and have equal number of rows.");

    float error = 0;
    for (size_t i = 0; i < y_true.numRows(); i++) {
        error += abs(y_true[i][0] - y_pred[i][0]);
    }
    return error / y_true.numRows();
}

float squared_euclidean_distance(const vector<float>& v1, const vector<float> v2){
    return squared_euclidean_distance(Matrix(v1), Matrix(v2));
}

float squared_euclidean_distance(const Matrix& v1, const Matrix& v2){
    assert(v1.numCols() == 1 && v2.numCols() == 1 && v1.numRows() == v2.numRows() && "Both vectors must be column vectors of the same size.");

    float sm=0;
    for(size_t i=0; i<v1.numRows(); i++){
        sm += pow(v1[i][0]-v2[i][0], 2);
    }

    return sm;
}

float mean(const vector<float>& v){
    float sm=0.0;
    for(int i=0; i<v.size(); i++){
        sm+=v[i];
    }
    return sm/((float)v.size());
}

float std_dev(const vector<float>& v){
    float mu=mean(v);
    float sm=0.0;
    for(int i=0; i<v.size(); i++){
        sm+=powf(v[i]-mu, 2.0);
    }
    return sqrt(((float)1.0/v.size())*sm);
}

Matrix normalize(const Matrix& m, const int axis){
    
    Matrix ret = Matrix(m.numRows(), m.numCols());

    // Row-wise normalize
    if(axis==0){

        for(int i=0; i<m.numRows(); i++){
            
            float mu=mean(m[i]);
            float sig=std_dev(m[i]);

            for(int j=0; j<m.numCols(); j++){
                ret[i][j]=(m[i][j]-mu)/sig;
            }
        }
    } 
    
    // Column-wise
    else if(axis==1){ 
        assert(false);
    } 
    
    // Normalize wrt entire matrix
    else {
        assert(false);
    }

    return ret;
}

Matrix softmax(const Matrix& m, const int axis){

    Matrix ret = Matrix(m.numRows(), m.numCols());

    // Row-wise softmax
    if (axis == 0) { // Row-wise softmax (each row sums to 1)
        for (int i = 0; i < m.numRows(); i++) {
            // Find max for numerical stability
            float max_val = m[i][0];
            for (int j = 1; j < m.numCols(); j++) {
                if (m[i][j] > max_val) max_val = m[i][j];
            }
            
            // Compute exponentials and sum
            float sum = 0.0f;
            for (int j = 0; j < m.numCols(); j++) {
                ret[i][j] = exp(m[i][j] - max_val); // Numerical stability
                sum += ret[i][j];
            }
            
            // Normalize
            for (int j = 0; j < m.numCols(); j++) {
                ret[i][j] /= sum;
            }
        }
    }
    
    // Column-wise
    else if(axis==1){ 
        assert(false);
    } 
    
    // Softmax wrt entire matrix
    else {
        assert(false);
    }

    return ret;
}

Matrix positional_encoder(const Matrix input_embeddings, int d_model) {
    Matrix ret = input_embeddings;
    
    for(int pos=0; pos<input_embeddings.numRows(); pos++){
        for(int i=0; i<input_embeddings.numCols(); i++){
            if(i%2==0){
                ret[pos][i]=sin(pos/powf(10000.0, (float)i/d_model));
            } else {
                ret[pos][i]=cos(pos/powf(10000.0, (float)(i-1)/d_model));
            }
        }
    }

    return ret;
}

Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V, int d_k, bool masked) {

    assert(Q.numCols() == K.numCols() && "Attention operation requires Q.numCols() == K.numCols() (key dimension)");

    Matrix scores = Q * K.T() / sqrt(d_k);

    if (masked) {
        for (int i = 0; i < scores.numRows(); ++i) {
            for (int j = 0; j < scores.numCols(); ++j) {
                if (i < j) { // Future tokens are masked
                    scores[i][j] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    return softmax(scores) * V;
}

Matrix add_and_norm(const Matrix& A, const Matrix& B){
    assert(A.numRows()==B.numRows()&&A.numCols()==B.numCols()&&"Add and norm operation requires A.numRows()==B.numRows()&&A.numCols()==B.numCols()");
    return normalize(A+B);
}

Matrix max(float n, const Matrix& m){
    Matrix ret = Matrix(m.numRows(), m.numCols());

    for(int i=0; i<m.numRows(); i++){
        for(int j=0; j<m.numCols(); j++){
            ret[i][j]=std::max(n, m[i][j]);  // make sure overloads properly
        }
    }

    return ret;
}

Matrix max(const Matrix& m, float n){
    return max(n, m);
}

}
