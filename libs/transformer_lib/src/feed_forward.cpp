#include <transformer_lib/feed_forward.h>
#include <math_lib/math_lib.h>

// Assume ReLU activation function
FeedForward::FeedForward(int d_model, int d_ff)
    : d_model(d_model),
      d_ff(d_ff),
      W_1(make_shared<Tensor>(Matrix(d_model, d_ff))),
      W_2(make_shared<Tensor>(Matrix(d_ff, d_model))),
      b_1(make_shared<Tensor>(Matrix(1, d_ff))),
      b_2(make_shared<Tensor>(Matrix(1, d_model))) {
    math_lib::xavier_uniform_initialization(W_1->data, d_model, d_ff);
    math_lib::xavier_uniform_initialization(W_2->data, d_ff, d_model);
    // Biases are initialized to zero by default
}

shared_ptr<Tensor> FeedForward::forward(const shared_ptr<Tensor>& x) const {
    // x: (batch, d_model)
    // W_1: (d_model, d_ff), b_1: (1, d_ff)
    // W_2: (d_ff, d_model), b_2: (1, d_model)
    auto h1 = math_lib::add(math_lib::matmul(x, W_1), b_1);
    auto h2 = math_lib::relu(h1);
    auto h3 = math_lib::add(math_lib::matmul(h2, W_2), b_2);
    return h3;
}

void FeedForward::step(float learning_rate) {
    // Update first layer weights and bias
    W_1->data = W_1->data - learning_rate * W_1->grad;
    b_1->data = b_1->data - learning_rate * b_1->grad;
    
    // Update second layer weights and bias
    W_2->data = W_2->data - learning_rate * W_2->grad;
    b_2->data = b_2->data - learning_rate * b_2->grad;
}

void FeedForward::zero_grad() {
    // Zero out gradients for first layer weights and bias
    W_1->grad = Matrix(W_1->grad.numRows(), W_1->grad.numCols());
    b_1->grad = Matrix(b_1->grad.numRows(), b_1->grad.numCols());
    
    // Zero out gradients for second layer weights and bias
    W_2->grad = Matrix(W_2->grad.numRows(), W_2->grad.numCols());
    b_2->grad = Matrix(b_2->grad.numRows(), b_2->grad.numCols());
}