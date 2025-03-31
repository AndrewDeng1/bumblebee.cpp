#include <transformer_lib/head.h>

Head::Head(int d_model, int d_k, int d_v, bool masked)
    : d_model(d_model),
      d_k(d_k),
      d_v(d_v),
      masked(masked),
      W_Q(d_model, d_k),
      W_K(d_model, d_k),
      W_V(d_model, d_v) {
    // Empty body

    math_lib::xavier_uniform_initialization(W_Q, d_model, d_k);
    math_lib::xavier_uniform_initialization(W_K, d_model, d_k);
    math_lib::xavier_uniform_initialization(W_V, d_model, d_v);
}

Matrix Head::forward(const Matrix& Q, const Matrix& K, const Matrix& V) const {

    // Better to have d_k passed in or just infer from dimension of K?
    //      - Requires attribute or to be passed into "forward" to pass in explicitly
    //      - Infer from dimension of K means don't need attributes
    return math_lib::attention(Q*W_Q, K*W_K, V*W_V, d_k, masked);
}