// test, clean up includes

#ifndef ENCODER_LAYER_H
#define ENCODER_LAYER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>
#include <transformer_lib/multi_head_attention.h>
#include <transformer_lib/feed_forward.h>

using namespace std;

class EncoderLayer {
    
    public:
        // Constructor
        EncoderLayer(int d_model, int d_ff, int h, int d_k, int d_v);
        
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
        
        MultiHeadAttention multi_head_attention;
        FeedForward feed_forward;
        shared_ptr<Tensor> W_Q, W_K, W_V;
};

#endif // ENCODER_LAYER_H