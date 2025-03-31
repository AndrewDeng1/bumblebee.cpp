#include <transformer_lib/feed_forward.h>

// Assume ReLU activation function
FeedForward::FeedForward(int d_model, int d_ff)
    : d_model(d_model),
      d_ff(d_ff),
      W_1(d_model, d_ff),
      W_2(d_ff, d_model),
      b_1(d_ff),
      b_2(d_model) {
    // Empty body

    math_lib::xavier_uniform_initialization(W_1, d_model, d_ff);
    math_lib::xavier_uniform_initialization(W_2, d_ff, d_model);
    
    // Bias vectors initialized to zero
}

Matrix FeedForward::forward(const Matrix& x) const {
    return math_lib::max(0, x*W_1+b_1)*W_2+b_2;
}