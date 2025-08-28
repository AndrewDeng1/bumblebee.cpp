#include <transformer_lib/encoder_layer.h>
#include <tensor_lib/tensor.h>

EncoderLayer::EncoderLayer(int d_model, int d_ff, int h, int d_k, int d_v)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      multi_head_attention(d_model, h, d_k, d_v),
      feed_forward(d_model, d_ff),
      W_Q(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_K(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_V(make_shared<Tensor>(Tensor({d_model, d_model}, true))) {
    tensor_lib::xavier_uniform_initialization(W_Q);
    tensor_lib::xavier_uniform_initialization(W_K);
    tensor_lib::xavier_uniform_initialization(W_V);
}

shared_ptr<Tensor> EncoderLayer::forward(const shared_ptr<Tensor>& X) const {
    auto Q = tensor_lib::matmul(X, W_Q);  // (seq_len × d_model)
    auto K = tensor_lib::matmul(X, W_K);  // (seq_len × d_model)
    auto V = tensor_lib::matmul(X, W_V);  // (seq_len × d_model)
    auto temp = tensor_lib::add_and_norm(X, multi_head_attention.forward(Q, K, V));
    return tensor_lib::add_and_norm(temp, feed_forward.forward(temp));
}

void EncoderLayer::step(float learning_rate) {
    // Update multi-head attention weights
    multi_head_attention.step(learning_rate);
    
    // Update feed-forward network weights
    feed_forward.step(learning_rate);
    
    // Update projection matrices
    W_Q->step(learning_rate);
    W_K->step(learning_rate);
    W_V->step(learning_rate);
}

void EncoderLayer::zero_grad() {
    // Zero out gradients in multi-head attention
    multi_head_attention.zero_grad();
    
    // Zero out gradients in feed-forward network
    feed_forward.zero_grad();
    
    // Zero out gradients for projection matrices
    W_Q->zero_grad();
    W_K->zero_grad();
    W_V->zero_grad();
}