#include <transformer_lib/encoder_layer.h>

EncoderLayer::EncoderLayer(int d_model, int d_ff, int h, int d_k, int d_v)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      multi_head_attention(d_model, h, d_k, d_v),
      feed_forward(d_model, d_ff),
      W_Q(d_model, d_model),
      W_K(d_model, d_model),
      W_V(d_model, d_model) {
    // Empty body
}

Matrix EncoderLayer::forward(const Matrix& X) const {
    Matrix Q=X*W_Q;
    Matrix K=X*W_K;
    Matrix V=X*W_V;
    Matrix temp=math_lib::add_and_norm(X, multi_head_attention.forward(Q, K, V));
    return math_lib::add_and_norm(temp, feed_forward.forward(temp));
}