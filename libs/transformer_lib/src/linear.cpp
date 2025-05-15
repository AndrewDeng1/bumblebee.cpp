#include <transformer_lib/linear.h>
#include <math_lib/math_lib.h>

Linear::Linear(int d_model, int V)
    : d_model(d_model),
      V(V),
      W(make_shared<Tensor>(Matrix(d_model, V))),
      b(make_shared<Tensor>(Matrix(1, V))) {
    // Xavier initialization for weights
    math_lib::xavier_uniform_initialization(W->data, d_model, V);
}

shared_ptr<Tensor> Linear::forward(const shared_ptr<Tensor>& X) const {
    // X: (batch, d_model), W: (d_model, V), b: (1, V)
    // Output: (batch, V)
    auto out = math_lib::add(math_lib::matmul(X, W), b);
    return out;
}

void Linear::step(float learning_rate) {
    // Update weights: W = W - learning_rate * dW
    W->data = W->data - learning_rate * W->grad;
    
    // Update bias: b = b - learning_rate * db
    b->data = b->data - learning_rate * b->grad;
}