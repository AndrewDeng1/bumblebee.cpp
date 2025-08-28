// test, clean up include

#ifndef HEAD_H
#define HEAD_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>

using namespace std;

class Head {
    
    public:
        // Constructor
        Head(int d_model, int d_k, int d_v, bool masked);
        
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
        int d_model, d_k, d_v;
        bool masked;
        
        // Weight matrices as tensors
        shared_ptr<Tensor> W_Q, W_K, W_V;
};

#endif // HEAD_H