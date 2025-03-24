#include "decoder_layer.h"

DecoderLayer::DecoderLayer(int d_model, int d_ff, int h, int d_k, int d_v) {
    this->d_model=d_model;
    this->d_ff=d_ff;
    this->h=h;
    this->d_k=d_k;
    this->d_v=d_v;

    this->multi_head_attention = MultiHeadAttention(d_model, h, d_k, d_v);
    this->masked_multi_head_attention = MultiHeadAttention(d_model, h, d_k, d_v, true);
    this->feed_forward = FeedForward(d_model, d_ff);
    this->W_Q_1 = Matrix(d_model, d_model);
    this->W_K_1 = Matrix(d_model, d_model);
    this->W_V_1 = Matrix(d_model, d_model);
    this->W_Q_2 = Matrix(d_model, d_model);
    this->W_K_2 = Matrix(d_model, d_model);
    this->W_V_2 = Matrix(d_model, d_model);
}

Matrix DecoderLayer::forward(const Matrix& X, const Matrix& encoder_out) const {
    Matrix Q_1=X*W_Q;
    Matrix K_1=X*W_K;
    Matrix V_1=X*W_V;
    Matrix temp_1=add_and_norm(X, multi_head_attention.forward(Q, K, V));
    Matrix temp_2=add_and_norm(temp_1, multi_head_attention.forward(Q, encoder_out, encoder_out));
    return add_and_norm(temp, feed_forward.forward(temp_2));
}