#include <transformer_lib/encoder.h>
#include <math_lib/math_lib.h>

Encoder::Encoder(int d_model, int d_ff, int h, int d_k, int d_v, int N)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      N(N),
      encoder_layers(N, EncoderLayer(d_model, d_ff, h, d_k, d_v)) {
    // Empty body
}

shared_ptr<Tensor> Encoder::forward(const shared_ptr<Tensor>& X) const {
    shared_ptr<Tensor> curr = X;
    for (int i = 0; i < encoder_layers.size(); i++) {
        curr = encoder_layers[i].forward(curr);
    }
    return curr;
}

void Encoder::step(float learning_rate) {
    // Update weights in each encoder layer
    for (auto& layer : encoder_layers) {
        layer.step(learning_rate);
    }
}