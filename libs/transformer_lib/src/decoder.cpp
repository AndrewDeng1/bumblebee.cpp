#include <transformer_lib/decoder.h>
#include <tensor_lib/tensor.h>

Decoder::Decoder(int d_model, int d_ff, int h, int d_k, int d_v, int N)
    : d_model(d_model),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      N(N),
      decoder_layers(N, DecoderLayer(d_model, d_ff, h, d_k, d_v)) {
    // Empty body
}

shared_ptr<Tensor> Decoder::forward(const shared_ptr<Tensor>& X, const shared_ptr<Tensor>& encoder_out) const {
    shared_ptr<Tensor> curr = X;
    for (int i = 0; i < decoder_layers.size(); i++) {
        curr = decoder_layers[i].forward(curr, encoder_out);
    }
    return curr;
}

void Decoder::zero_grad() {
    for (auto& layer : decoder_layers) {
        layer.zero_grad();
    }
}

void Decoder::step(float learning_rate) {
    for (auto& layer : decoder_layers) {
        layer.step(learning_rate);
    }
}