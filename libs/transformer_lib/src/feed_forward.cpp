#include <transformer_lib/feed_forward.h>
#include <tensor_lib/tensor.h>

// Assume ReLU activation function
FeedForward::FeedForward(int d_model, int d_ff)
    : d_model(d_model),
      d_ff(d_ff),
      W_1(make_shared<Tensor>(Matrix(d_model, d_ff))),
      W_2(make_shared<Tensor>(Matrix(d_ff, d_model))),
      b_1(make_shared<Tensor>(Matrix(1, d_ff))),
      b_2(make_shared<Tensor>(Matrix(1, d_model))) {
    tensor_lib::xavier_uniform_initialization(W_1);
    tensor_lib::xavier_uniform_initialization(W_2);
    // Biases are initialized to zero by default
}

shared_ptr<Tensor> FeedForward::forward(const shared_ptr<Tensor>& x) const {
    // x: (batch, d_model)
    // W_1: (d_model, d_ff), b_1: (1, d_ff)
    // W_2: (d_ff, d_model), b_2: (1, d_model)
    auto h1 = tensor_lib::add(tensor_lib::matmul(x, W_1), b_1);
    auto h2 = tensor_lib::relu(h1);
    auto h3 = tensor_lib::add(tensor_lib::matmul(h2, W_2), b_2);
    return h3;
}

void FeedForward::step(float learning_rate) {
    // Update first layer weights and bias
    W_1->step(learning_rate);
    b_1->step(learning_rate);
    
    // Update second layer weights and bias
    W_2->step(learning_rate);
    b_2->step(learning_rate);
}

void FeedForward::zero_grad() {
    // Zero out gradients for first layer weights and bias
    W_1->zero_grad();
    b_1->zero_grad();
    
    // Zero out gradients for second layer weights and bias
    W_2->zero_grad();
    b_2->zero_grad();
}