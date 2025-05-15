#include <transformer_lib/encoder_layer.h>
#include <math_lib/math_lib.h>

EncoderLayer::EncoderLayer(int d_model, int d_ff, int h, int d_k, int d_v)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      multi_head_attention(d_model, h, d_k, d_v),
      feed_forward(d_model, d_ff),
      W_Q(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_K(make_shared<Tensor>(Matrix(d_model, d_model))),
      W_V(make_shared<Tensor>(Matrix(d_model, d_model))) {
    math_lib::xavier_uniform_initialization(W_Q->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_K->data, d_model, d_model);
    math_lib::xavier_uniform_initialization(W_V->data, d_model, d_model);
}

shared_ptr<Tensor> EncoderLayer::forward(const shared_ptr<Tensor>& X) const {
    auto Q = math_lib::matmul(X, W_Q);  // (seq_len × d_model)
    auto K = math_lib::matmul(X, W_K);  // (seq_len × d_model)
    auto V = math_lib::matmul(X, W_V);  // (seq_len × d_model)
    auto temp = math_lib::add_and_norm(X, multi_head_attention.forward(Q, K, V));
    return math_lib::add_and_norm(temp, feed_forward.forward(temp));
}

void EncoderLayer::step(float learning_rate) {
    // Update multi-head attention weights
    multi_head_attention.step(learning_rate);
    
    // Update feed-forward network weights
    feed_forward.step(learning_rate);
    
    // Update projection matrices
    W_Q->data = W_Q->data - learning_rate * W_Q->grad;
    W_K->data = W_K->data - learning_rate * W_K->grad;
    W_V->data = W_V->data - learning_rate * W_V->grad;
}

void EncoderLayer::zero_grad() {
    // Zero out gradients in multi-head attention
    multi_head_attention.zero_grad();
    
    // Zero out gradients in feed-forward network
    feed_forward.zero_grad();
    
    // Zero out gradients for projection matrices
    W_Q->grad = Matrix(W_Q->grad.numRows(), W_Q->grad.numCols());
    W_K->grad = Matrix(W_K->grad.numRows(), W_K->grad.numCols());
    W_V->grad = Matrix(W_V->grad.numRows(), W_V->grad.numCols());
}