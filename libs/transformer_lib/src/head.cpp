#include <transformer_lib/head.h>
#include <tensor_lib/tensor.h>

Head::Head(int d_model, int d_k, int d_v, bool masked)
    : d_model(d_model),
      d_k(d_k),
      d_v(d_v),
      masked(masked),
      W_Q(make_shared<Tensor>(Matrix(d_model, d_k))),
      W_K(make_shared<Tensor>(Matrix(d_model, d_k))),
      W_V(make_shared<Tensor>(Matrix(d_model, d_v))) {
    tensor_lib::xavier_uniform_initialization(W_Q);
    tensor_lib::xavier_uniform_initialization(W_K);
    tensor_lib::xavier_uniform_initialization(W_V);
}

shared_ptr<Tensor> Head::forward(const shared_ptr<Tensor>& Q, const shared_ptr<Tensor>& K, const shared_ptr<Tensor>& V) const {
    // Q, K, V: (batch, d_model)
    // W_Q, W_K, W_V: (d_model, d_k) or (d_model, d_v)
    auto Q_proj = tensor_lib::matmul(Q, W_Q);
    auto K_proj = tensor_lib::matmul(K, W_K);
    auto V_proj = tensor_lib::matmul(V, W_V);
    return tensor_lib::attention(Q_proj, K_proj, V_proj, d_k, masked);
}

void Head::zero_grad() {
    // Zero out gradients for query, key, and value weight matrices
    W_Q->zero_grad();
    W_K->zero_grad();
    W_V->zero_grad();
}

void Head::step(float learning_rate) {
    W_Q->step(learning_rate);
    W_K->step(learning_rate);
    W_V->step(learning_rate);
}