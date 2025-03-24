#include "multi_head_attention.h"

MultiHeadAttention::MultiHeadAttention(int d_model, int h, int d_k, int d_v, bool masked)
    : d_model(d_model),
      h(h),
      heads(h, Head(d_model, d_k, d_v, masked)),
      masked(masked),
      W_O(h * d_v, d_model) {
    // Empty body
}

Matrix MultiHeadAttention::forward(Matrix& Q, Matrix& K, Matrix& V) const {
    Matrix m=heads[0].forward(Q, K, V);
    for(int i=1; i<h; i++){
        m=m.concat(heads[i].forward(Q, K, V));
    }
    return m*W_O;
}