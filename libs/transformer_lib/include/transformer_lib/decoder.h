// test, clean up includes

#ifndef DECODER_H
#define DECODER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>
#include <transformer_lib/decoder_layer.h>

using namespace std;

class Decoder {
    
    public:
        // Constructor
        Decoder(int d_model, int d_ff, int h, int d_k, int d_v, int N);
        
        // Forward pass
        shared_ptr<Tensor> forward(const shared_ptr<Tensor>& X, const shared_ptr<Tensor>& encoder_out) const;
        
        // Zero gradients
        void zero_grad();
        
        // Update weights
        void step(float learning_rate);

    private:
        int d_model;
        int d_ff;
        int h;
        int d_k;
        int d_v;
        int N;
        
        vector<DecoderLayer> decoder_layers;
};

#endif // DECODER_H