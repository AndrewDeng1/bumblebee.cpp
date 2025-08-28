#include <transformer_lib/linear.h>
#include <tensor_lib/tensor.h>

Linear::Linear(int d_model, int V)
    : d_model(d_model),
      V(V),
      W(make_shared<Tensor>(Tensor({d_model, V}, true))),
      b(make_shared<Tensor>(Tensor({V, 1}, true))) {
    // Xavier initialization for weights
    tensor_lib::xavier_uniform_initialization(W);
}

shared_ptr<Tensor> Linear::forward(const shared_ptr<Tensor>& X) const {
    // X: (batch, d_model), W: (d_model, V), b: (1, V)
    // Output: (batch, V)
    auto out = tensor_lib::matmul(X, W)+b;
    return out;
}

void Linear::zero_grad() {
    W->zero_grad();
    b->zero_grad();
}

void Linear::step(float learning_rate) {
    // Update weights: W = W - learning_rate * dW
    W->step(learning_rate);
    
    // Update bias: b = b - learning_rate * db
    b->step(learning_rate);
}