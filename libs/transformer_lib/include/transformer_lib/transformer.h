// test, clean up includes

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <math_lib/math_lib.h>
#include <transformer_lib/encoder.h>
#include <transformer_lib/decoder.h>
#include <transformer_lib/linear.h>

using namespace std;

class Transformer {
    
    public:
        // Constructor
        Transformer(float learning_rate, int d_model, int V, int d_ff, int h, int d_k, int d_v, int N);
        
        // Forward pass: takes input and output token sequences
        shared_ptr<Tensor> forward(
            const vector<string>& input_tokens,    // Input token sequence
            const vector<string>& output_tokens    // Output token sequence
        ) const;
        
        // Zero gradients
        void zero_grad();
        
        // Backward pass: takes predictions and target one-hot encodings
        void backward(
            const shared_ptr<Tensor>& predictions,  // Shape: (seq_len, V)
            const shared_ptr<Tensor>& targets       // Shape: (seq_len, V)
        );
        
        // Update weights
        void step();

        // Compute cross entropy loss between predictions and targets
        shared_ptr<Tensor> cross_entropy_loss(
            const shared_ptr<Tensor>& predictions,  // Shape: (seq_len, V)
            const shared_ptr<Tensor>& targets       // Shape: (seq_len, V)
        ) const;

    private:
        float learning_rate;
        int d_model;
        int V;
        int d_ff;
        int h;
        int d_k;
        int d_v;
        int N;

        Encoder encoder;
        Decoder decoder;
        Linear linear;
        
        // Token embedding layer and its gradients
        shared_ptr<unordered_map<string, vector<float>>> token_embeddings;
        shared_ptr<unordered_map<string, vector<float>>> token_embedding_grads;
};

#endif // TRANSFORMER_H
