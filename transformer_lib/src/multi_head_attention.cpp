#include "multi_head_attention.h"

MultiHeadAttention::MultiHeadAttention(int d_model, int h, int d_k, int d_v, bool masked=false) {
    this->d_model=d_model;
    this->h=h;
    this->heads=vector<Head>(h, masked);
    this->masked=masked;
    this->W_O=Matrix(h*d_v, d_model);
}

Matrix MultiHeadAttention::forward(Matrix& X, Matrix& Q, Matrix& K, Matrix& V) const {
    Matrix m=heads[0].forward(Q, K, V);
    for(int i=1; i<h; i++){
        m=m.concat(heads[i].forward(Q, K, V));
    }
    return m*W_O;
}