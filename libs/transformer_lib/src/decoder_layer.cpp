#include <transformer_lib/decoder_layer.h>

DecoderLayer::DecoderLayer(int d_model, int d_ff, int h, int d_k, int d_v)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      multi_head_attention(d_model, h, d_k, d_v),
      masked_multi_head_attention(d_model, h, d_k, d_v, true),
      feed_forward(d_model, d_ff),
      W_Q_1(d_model, d_model),
      W_K_1(d_model, d_model),
      W_V_1(d_model, d_model),
      W_Q_2(d_model, d_model),
      W_K_2(d_model, d_model),
      W_V_2(d_model, d_model) {
    // Empty body

    math_lib::xavier_uniform_initialization(W_Q_1, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_K_1, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_V_1, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_Q_2, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_K_2, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_V_2, d_model, d_model);
}

Matrix DecoderLayer::forward(const Matrix& X, const Matrix& encoder_out) const {
    Matrix Q_1=X*W_Q_1;
    Matrix K_1=X*W_K_1;
    Matrix V_1=X*W_V_1;
    Matrix temp_1=math_lib::add_and_norm(X, multi_head_attention.forward(Q_1, K_1, V_1));

    Matrix Q_2=temp_1*W_Q_2;
    Matrix K_2=encoder_out*W_K_2;
    Matrix V_2=encoder_out*W_V_2;

    Matrix temp_2=math_lib::add_and_norm(temp_1, multi_head_attention.forward(Q_2, K_2, V_2));
    return math_lib::add_and_norm(temp_2, feed_forward.forward(temp_2));
}