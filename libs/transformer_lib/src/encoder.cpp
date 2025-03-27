#include <transformer_lib/encoder.h>

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

Matrix Encoder::forward(const Matrix& X) const {
    
    Matrix curr=X;

    for(int i=0; i<encoder_layers.size(); i++){
        curr=encoder_layers[i].forward(curr);
    }
    
    return curr;
}