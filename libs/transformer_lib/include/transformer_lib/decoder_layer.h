// test, clean up includes

#ifndef DECODER_LAYER_H
#define DECODER_LAYER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>
#include <transformer_lib/decoder_layer.h>
#include <transformer_lib/multi_head_attention.h>
#include <transformer_lib/feed_forward.h>

using namespace std;

class DecoderLayer {
    
    public:
        // Constructor
        DecoderLayer(int d_model, int d_ff, int h, int d_k, int d_v);
        
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
        
        MultiHeadAttention multi_head_attention, masked_multi_head_attention;
        FeedForward feed_forward;
        shared_ptr<Tensor> W_Q_1, W_K_1, W_V_1, W_Q_2, W_K_2, W_V_2;
};

#endif // DECODER_LAYER_H