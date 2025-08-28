#include <transformer_lib/multi_head_attention.h>
#include <tensor_lib/tensor.h>

MultiHeadAttention::MultiHeadAttention(int d_model, int h, int d_k, int d_v, bool masked)
    : d_model(d_model),
      h(h),
      heads(h, Head(d_model, d_k, d_v, masked)),
      masked(masked),
      W_O(make_shared<Tensor>(Tensor({h * d_v, d_model}, true))) {
    tensor_lib::xavier_uniform_initialization(W_O);
}

shared_ptr<Tensor> MultiHeadAttention::forward(const shared_ptr<Tensor>& Q, const shared_ptr<Tensor>& K, const shared_ptr<Tensor>& V) const {
    // Concatenate the outputs of all heads
    std::vector<shared_ptr<Tensor>> head_outputs;
    for (int i = 0; i < h; ++i) {
        head_outputs.push_back(heads[i].forward(Q, K, V));
    }
    auto m = tensor_lib::concat(head_outputs, 1); // Concatenate along columns
    return tensor_lib::matmul(m, W_O);
}

void MultiHeadAttention::step(float learning_rate) {
    // Update output projection matrix
    W_O->step(learning_rate);
    
    // Update weights in each attention head
    for (auto& head : heads) {
        head.step(learning_rate);
    }
}

void MultiHeadAttention::zero_grad() {
    // Zero out gradient for output projection matrix
    W_O->zero_grad();
    
    // Zero out gradients in each attention head
    for (auto& head : heads) {
        head.zero_grad();
    }
}