#include <transformer_lib/decoder_layer.h>
#include <tensor_lib/tensor.h>

DecoderLayer::DecoderLayer(int d_model, int d_ff, int h, int d_k, int d_v)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      multi_head_attention(d_model, h, d_k, d_v),
      masked_multi_head_attention(d_model, h, d_k, d_v, true),
      feed_forward(d_model, d_ff),
      W_Q_1(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_K_1(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_V_1(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_Q_2(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_K_2(make_shared<Tensor>(Tensor({d_model, d_model}, true))),
      W_V_2(make_shared<Tensor>(Tensor({d_model, d_model}, true))) {
    tensor_lib::xavier_uniform_initialization(W_Q_1);
    tensor_lib::xavier_uniform_initialization(W_K_1);
    tensor_lib::xavier_uniform_initialization(W_V_1);
    tensor_lib::xavier_uniform_initialization(W_Q_2);
    tensor_lib::xavier_uniform_initialization(W_K_2);
    tensor_lib::xavier_uniform_initialization(W_V_2);
}

shared_ptr<Tensor> DecoderLayer::forward(const shared_ptr<Tensor>& X, const shared_ptr<Tensor>& encoder_out) const {
    auto Q_1 = tensor_lib::matmul(X, W_Q_1);
    auto K_1 = tensor_lib::matmul(X, W_K_1);
    auto V_1 = tensor_lib::matmul(X, W_V_1);
    auto temp_1 = tensor_lib::add_and_norm(X, multi_head_attention.forward(Q_1, K_1, V_1));

    auto Q_2 = tensor_lib::matmul(temp_1, W_Q_2);
    auto K_2 = tensor_lib::matmul(encoder_out, W_K_2);
    auto V_2 = tensor_lib::matmul(encoder_out, W_V_2);

    auto temp_2 = tensor_lib::add_and_norm(temp_1, multi_head_attention.forward(Q_2, K_2, V_2));
    return tensor_lib::add_and_norm(temp_2, feed_forward.forward(temp_2));
}

void DecoderLayer::step(float learning_rate) {
    // Update masked multi-head attention weights
    masked_multi_head_attention.step(learning_rate);
    
    // Update cross-attention multi-head attention weights
    multi_head_attention.step(learning_rate);
    
    // Update feed-forward network weights
    feed_forward.step(learning_rate);
    
    // // Update projection matrices for masked attention
    W_Q_1->step(learning_rate);
    W_K_1->step(learning_rate);
    W_V_1->step(learning_rate);
    W_Q_2->step(learning_rate);
    W_K_2->step(learning_rate);
    W_V_2->step(learning_rate);
}

void DecoderLayer::zero_grad() {
    // Zero out gradients in masked multi-head attention
    masked_multi_head_attention.zero_grad();
    
    // Zero out gradients in cross-attention multi-head attention
    multi_head_attention.zero_grad();
    
    // Zero out gradients in feed-forward network
    feed_forward.zero_grad();
    
    // // Zero out gradients for projection matrices (masked attention)
    W_Q_1->zero_grad();
    W_K_1->zero_grad();
    W_V_1->zero_grad();
    W_Q_2->zero_grad();
    W_K_2->zero_grad();
    W_V_2->zero_grad();
}