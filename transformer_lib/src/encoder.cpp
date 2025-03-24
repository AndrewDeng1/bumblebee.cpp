#include "encoder.h"

Encoder::Encoder(int d_model, int d_ff, int h, int d_k, int d_v, int N) {
    this->d_model=d_model;
    this->d_ff=d_ff;
    this->h=h;
    this->d_k=d_k;
    this->d_v=d_v;
    this->N=N;
    
    this->encoder_layers=vector<EncoderLayer>(N, EncoderLayer(d_model, d_ff, h, d_k, d_v));
}

Matrix Encoder::forward(Matrix& X) const {
    
    Matrix curr=X;

    for(int i=0; i<encoder_layers.size(); i++){
        curr=encoder_layers[i].forward(curr);
    }
    
    return curr;
}