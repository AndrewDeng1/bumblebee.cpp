#include "feed_forward.h"

// Assume ReLU activation function
FeedForward::FeedForward(int d_model, int d_ff) {
    this->d_model=d_model;
    this->d_ff=d_ff;
    this->W_1=Matrix(d_model, d_ff);
    this->W_2=Matrix(d_model, d_ff);
    this->b_1=vector<float>(d_ff);
    this->b_2=vector<float>(d_ff);
}

Matrix FeedForward::forward(Matrix& x) const {
    return max(0, x*W_1+b_1)*W_2+b_2;
}