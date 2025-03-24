#include "feed_forward.h"

// Assume ReLU activation function
FeedForward::FeedForward(int d_model, int d_ff)
    : d_model(d_model),
      d_ff(d_ff),
      W_1(d_model, d_ff),
      W_2(d_model, d_ff),
      b_1(d_ff),
      b_2(d_ff) {
    // Empty body
}

Matrix FeedForward::forward(const Matrix& x) const {
    return math_lib::max(0, x*W_1+b_1)*W_2+b_2;
}