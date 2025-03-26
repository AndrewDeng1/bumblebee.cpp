#include <transformer_lib/decoder.h>

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

Matrix Decoder::forward(const Matrix& X, const Matrix& encoder_out) const {
    
    Matrix curr=X;

    for(int i=0; i<decoder_layers.size(); i++){
        curr=decoder_layers[i].forward(curr, encoder_out);
    }
    
    return curr;
}