#include "decoder.h"

Decoder::Decoder(int d_model, int d_ff, int h, int d_k, int d_v, int N) {
    this->d_model=d_model;
    this->d_ff=d_ff;
    this->h=h;
    this->d_k=d_k;
    this->d_v=d_v;
    this->N=N;
    
    this->decoder_layers=vector<DecoderLayer>(N, DecoderLayer(d_model, d_ff, h, d_k, d_v));
}

Matrix Decoder::forward(const Matrix& X, const Matrix& encoder_out) const {
    
    Matrix curr=X;

    for(int i=0; i<decoder_layers.size(); i++){
        curr=decoder_layers[i].forward(curr, encoder_out);
    }
    
    return curr;
}