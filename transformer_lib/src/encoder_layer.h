// test, clean up includes

#ifndef ENCODER_LAYER_H
#define ENCODER_LAYER_H

#include <vector>
#include <iostream>
#include <cassert>
#include "../../math_lib/src/math_lib.h"
#include "multi_head_attention.h"
#include "feed_forward.h"

using namespace std;

class EncoderLayer {
    
    public:

        // Declare signature of constructor methods
        EncoderLayer(int d_model, int d_ff, int h, int d_k, int d_v);
        Matrix forward(const Matrix& X) const;
    
    private:

        int d_model;
        int d_ff;
        int h;
        int d_k;
        int d_v;
        
        MultiHeadAttention multi_head_attention;
        FeedForward feed_forward;
        Matrix W_Q, W_K, W_V;
};

#endif // ENCODER_LAYER_H