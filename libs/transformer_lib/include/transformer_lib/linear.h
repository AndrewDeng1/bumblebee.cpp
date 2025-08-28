// test, clean up includes

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <math_lib/math_lib.h>

using namespace std;

class Linear {
    
    public:
        // Constructor
        Linear(int d_model, int V);
        
        // Forward pass
        shared_ptr<Tensor> forward(const shared_ptr<Tensor>& X) const;
        
        // Zero gradients
        void zero_grad();
        
        // Update weights
        void step(float learning_rate);
    
    private:
        int d_model;
        int V;

        // Weights and bias as tensors
        shared_ptr<Tensor> W;
        shared_ptr<Tensor> b;
};

#endif // LINEAR_LAYER_H