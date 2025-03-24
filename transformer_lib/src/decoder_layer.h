// test, clean up includes

#ifndef DECODER_LAYER_H
#define DECODER_LAYER_H

#include <vector>
#include <iostream>
#include <cassert>
#include "../../math_lib/src/math_lib.h"
#include "decoder_layer.h"
#include "multi_head_attention.h"
#include "feed_forward.h"

using namespace std;

class DecoderLayer {
    
    public:

        // Declare signature of constructor methods
        DecoderLayer(int d_model, int d_ff, int h, int d_k, int d_v);
        Matrix forward(const Matrix& X, const Matrix& encoder_out) const;
    
    private:

        int d_model;
        int d_ff;
        int h;
        int d_k;
        int d_v;
        
        MultiHeadAttention multi_head_attention, masked_multi_head_attention;
        FeedForward feed_forward;
        Matrix W_Q_1, W_K_1, W_V_1, W_Q_2, W_K_2, W_V_2;
};

#endif // DECODER_LAYER_H