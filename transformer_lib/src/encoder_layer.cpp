#include "encoder_layer.h"

EncoderLayer::EncoderLayer(int d_model, int d_ff, int h, int d_k, int d_v) {
    this->d_model=d_model;
    this->d_ff=d_ff;
    this->h=h;
    this->d_k=d_k;
    this->d_v=d_v;

    this->multi_head_attention = MultiHeadAttention(d_model, h, d_k, d_v);
    this->feed_forward = FeedForward(d_model, d_ff);
    this->W_Q = Matrix(d_model, d_model);
    this->W_K = Matrix(d_model, d_model);
    this->W_V = Matrix(d_model, d_model);
}

Matrix EncoderLayer::forward(Matrix& X) const {
    Matrix Q=X*W_Q;
    Matrix K=X*W_K;
    Matrix V=X*W_V;
    Matrix temp=add_and_norm(X, multi_head_attention.forward(Q, K, V));
    return add_and_norm(temp, feed_forward.forward(temp));
}