// test, clean up includes

#ifndef ENCODER_H
#define ENCODER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>
#include <transformer_lib/encoder_layer.h>

using namespace std;

class Encoder {
    
    public:
        // Constructor
        Encoder(int d_model, int d_ff, int h, int d_k, int d_v, int N);
        
        // Forward pass
        shared_ptr<Tensor> forward(const shared_ptr<Tensor>& X) const;
        
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
        
        vector<EncoderLayer> encoder_layers;
};

#endif // ENCODER_H