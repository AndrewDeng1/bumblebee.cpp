#include <transformer_lib/decoder_layer.h>
#include <math_lib/math_lib.h>

DecoderLayer::DecoderLayer(int d_model, int d_ff, int h, int d_k, int d_v)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      multi_head_attention(d_model, h, d_k, d_v),
      masked_multi_head_attention(d_model, h, d_k, d_v, true),
      feed_forward(d_model, d_ff),
      W_Q_1(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_K_1(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_V_1(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_Q_2(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_K_2(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_V_2(make_shared<Tensor>(Matrix(d_model, d_model))) {
    math_lib::xavier_uniform_initialization(W_Q_1->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_K_1->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_V_1->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_Q_2->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_K_2->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_V_2->data, d_model, d_model);
}

shared_ptr<Tensor> DecoderLayer::forward(const shared_ptr<Tensor>& X, const shared_ptr<Tensor>& encoder_out) const {
    auto Q_1 = math_lib::matmul(X, W_Q_1);
    auto K_1 = math_lib::matmul(X, W_K_1);
    auto V_1 = math_lib::matmul(X, W_V_1);
    auto temp_1 = math_lib::add_and_norm(X, multi_head_attention.forward(Q_1, K_1, V_1));

    auto Q_2 = math_lib::matmul(temp_1, W_Q_2);
    auto K_2 = math_lib::matmul(encoder_out, W_K_2);
    auto V_2 = math_lib::matmul(encoder_out, W_V_2);

    auto temp_2 = math_lib::add_and_norm(temp_1, multi_head_attention.forward(Q_2, K_2, V_2));
    return math_lib::add_and_norm(temp_2, feed_forward.forward(temp_2));
}

void DecoderLayer::step(float learning_rate) {
    // Update masked multi-head attention weights
    masked_multi_head_attention.step(learning_rate);
    
    // Update cross-attention multi-head attention weights
    multi_head_attention.step(learning_rate);
    
    // Update feed-forward network weights
    feed_forward.step(learning_rate);
    
    // Update projection matrices for masked attention
    W_Q_1->data = W_Q_1->data - learning_rate * W_Q_1->grad;
    W_K_1->data = W_K_1->data - learning_rate * W_K_1->grad;
    W_V_1->data = W_V_1->data - learning_rate * W_V_1->grad;
    
    // Update projection matrices for cross attention
    W_Q_2->data = W_Q_2->data - learning_rate * W_Q_2->grad;
    W_K_2->data = W_K_2->data - learning_rate * W_K_2->grad;
    W_V_2->data = W_V_2->data - learning_rate * W_V_2->grad;
}

void DecoderLayer::zero_grad() {
    // Zero out gradients in masked multi-head attention
    masked_multi_head_attention.zero_grad();
    
    // Zero out gradients in cross-attention multi-head attention
    multi_head_attention.zero_grad();
    
    // Zero out gradients in feed-forward network
    feed_forward.zero_grad();
    
    // Zero out gradients for projection matrices (masked attention)
    W_Q_1->grad = Matrix(W_Q_1->grad.numRows(), W_Q_1->grad.numCols());
    W_K_1->grad = Matrix(W_K_1->grad.numRows(), W_K_1->grad.numCols());
    W_V_1->grad = Matrix(W_V_1->grad.numRows(), W_V_1->grad.numCols());
    
    // Zero out gradients for projection matrices (cross attention)
    W_Q_2->grad = Matrix(W_Q_2->grad.numRows(), W_Q_2->grad.numCols());
    W_K_2->grad = Matrix(W_K_2->grad.numRows(), W_K_2->grad.numCols());
    W_V_2->grad = Matrix(W_V_2->grad.numRows(), W_V_2->grad.numCols());
}