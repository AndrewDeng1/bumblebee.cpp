#include <transformer_lib/linear.h>

Linear::Linear(int d_model, int V)
    : d_model(d_model),
      V(V),
      W(d_model, V),
      b(V) {
    // Empty body
}

Matrix Linear::forward(const Matrix& X) const {
    return X*W+b;
}