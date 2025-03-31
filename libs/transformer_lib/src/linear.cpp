#include <transformer_lib/linear.h>

Linear::Linear(int d_model, int V)
    : d_model(d_model),
      V(V),
      W(d_model, V),
      b(V) {
    // Empty body

    math_lib::xavier_uniform_initialization(W, d_model, V);
}

Matrix Linear::forward(const Matrix& X) const {
    return X*W+b;
}