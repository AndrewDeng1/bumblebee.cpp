// test, clean up includes

#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>

using namespace std;

class FeedForward {
    
    public:
        // Constructor
        FeedForward(int d_model, int d_ff);
        
        // Forward pass
        shared_ptr<Tensor> forward(const shared_ptr<Tensor>& X) const;
        
        // Zero gradients
        void zero_grad();
        
        // Update weights
        void step(float learning_rate);
    
    private:
        int d_model, d_ff;

        // Weight matrices and biases as tensors
        shared_ptr<Tensor> W_1, W_2;
        shared_ptr<Tensor> b_1, b_2;
};

#endif // FEED_FORWARD_H