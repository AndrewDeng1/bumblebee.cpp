// test, clean up include

#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>
#include <transformer_lib/head.h>

using namespace std;

class MultiHeadAttention {
    
    public:
        // Constructor
        MultiHeadAttention(int d_model, int h, int d_k, int d_v, bool masked=false);
        
        // Forward pass
        shared_ptr<Tensor> forward(
            const shared_ptr<Tensor>& Q,
            const shared_ptr<Tensor>& K,
            const shared_ptr<Tensor>& V
        ) const;
        
        // Zero gradients
        void zero_grad();
        
        // Update weights
        void step(float learning_rate);
    
    private:
        int d_model;
        int h;
        bool masked;

        vector<Head> heads;
        shared_ptr<Tensor> W_O;
};

#endif // MULTI_HEAD_ATTENTION_H