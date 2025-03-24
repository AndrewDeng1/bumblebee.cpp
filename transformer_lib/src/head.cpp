#include "head.h"

Head::Head(int d_model, int d_k, int d_v, bool masked) {
    
    this->d_model=d_model;
    this->d_k=d_k;
    this->d_v=d_v;
    this->masked=masked;

    this->W_Q=Matrix(d_model, d_k);
    this->W_K=Matrix(d_model, d_k);
    this->W_V=Matrix(d_model, d_v);
}

Matrix Head::forward(Matrix& X, Matrix& Q, Matrix& K, Matrix& V) const {
    return attention(Q*W_Q, K*W_K, V*W_V, masked);
}