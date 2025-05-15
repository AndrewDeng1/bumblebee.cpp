#include <transformer_lib/head.h>
#include <math_lib/math_lib.h>

Head::Head(int d_model, int d_k, int d_v, bool masked)
    : d_model(d_model),
      d_k(d_k),
      d_v(d_v),
      masked(masked),
      W_Q(make_shared<Tensor>(Matrix(d_model, d_k))),
      W_K(make_shared<Tensor>(Matrix(d_model, d_k))),
      W_V(make_shared<Tensor>(Matrix(d_model, d_v))) {
    math_lib::xavier_uniform_initialization(W_Q->data, d_model, d_k);
    math_lib::xavier_uniform_initialization(W_K->data, d_model, d_k);
    math_lib::xavier_uniform_initialization(W_V->data, d_model, d_v);
}

shared_ptr<Tensor> Head::forward(const shared_ptr<Tensor>& Q, const shared_ptr<Tensor>& K, const shared_ptr<Tensor>& V) const {
    // Q, K, V: (batch, d_model)
    // W_Q, W_K, W_V: (d_model, d_k) or (d_model, d_v)
    auto Q_proj = math_lib::matmul(Q, W_Q);
    auto K_proj = math_lib::matmul(K, W_K);
    auto V_proj = math_lib::matmul(V, W_V);
    return math_lib::attention(Q_proj, K_proj, V_proj, d_k, masked);
}

void Head::zero_grad() {
    // Zero out gradients for query, key, and value weight matrices
    W_Q->grad = Matrix(W_Q->grad.numRows(), W_Q->grad.numCols());
    W_K->grad = Matrix(W_K->grad.numRows(), W_K->grad.numCols());
    W_V->grad = Matrix(W_V->grad.numRows(), W_V->grad.numCols());
}