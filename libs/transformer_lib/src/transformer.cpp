#include <transformer_lib/transformer.h>
#include <math_lib/math_lib.h>
#include <unordered_map>

Transformer::Transformer(float learning_rate, int d_model, int V, int d_ff, int h, int d_k, int d_v, int N)
    : learning_rate(learning_rate),
      d_model(d_model),
      V(V),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      N(N),
      encoder(d_model, d_ff, h, d_k, d_v, N),
      decoder(d_model, d_ff, h, d_k, d_v, N),
      linear(d_model, V) {
    // Initialize token embeddings map and its gradients
    token_embeddings = make_shared<unordered_map<string, vector<float>>>();
    token_embedding_grads = make_shared<unordered_map<string, vector<float>>>();
}

shared_ptr<Tensor> Transformer::forward(
    const vector<string>& input_tokens,
    const vector<string>& output_tokens
) const {
    // Get embeddings for input and output sequences
    auto input_embeddings = math_lib::embed(input_tokens, *token_embeddings, d_model);
    auto output_embeddings = math_lib::embed(output_tokens, *token_embeddings, d_model);
    
    // Add positional encoding
    input_embeddings->data = math_lib::positional_encoder(input_embeddings->data, d_model);
    output_embeddings->data = math_lib::positional_encoder(output_embeddings->data, d_model);
    
    // Forward pass through encoder
    auto encoder_out = encoder.forward(input_embeddings);
    
    // Forward pass through decoder
    auto decoder_out = decoder.forward(output_embeddings, encoder_out);
    
    // Final linear layer and softmax
    return math_lib::softmax(linear.forward(decoder_out));
}

// targets.requires_grad = false, but doesn't rly matter
shared_ptr<Tensor> Transformer::cross_entropy_loss(
    const shared_ptr<Tensor>& predictions,
    const shared_ptr<Tensor>& targets
) const {
    int seq_len = predictions->data.numRows();
    int vocab_size = predictions->data.numCols();
    
    // Create a new tensor for the loss
    auto loss_tensor = make_shared<Tensor>(Matrix(1, 1), true);  // Single scalar value
    float total_loss = 0.0f;
    
    // Compute cross entropy loss
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < vocab_size; j++) {
            if (targets->data[i][j] > 0) {  // Only compute loss for the target token
                float pred = predictions->data[i][j];
                float target = targets->data[i][j];
                total_loss -= target * log(pred + 1e-10f);  // Add small epsilon for numerical stability
            }
        }
    }
    
    // Set the loss value
    loss_tensor->data[0][0] = total_loss;
    
    // Set up the backward function
    loss_tensor->backward_fn = [predictions, targets]() {
        int seq_len = predictions->data.numRows();
        int vocab_size = predictions->data.numCols();
        Matrix loss_grad(seq_len, vocab_size);
        
        // Compute gradient: prediction - target
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < vocab_size; j++) {
                if (targets->data[i][j] > 0) {
                    loss_grad[i][j] = predictions->data[i][j] - targets->data[i][j];
                }
            }
        }
        
        // Set gradients for parent tensors
        if (predictions->requires_grad) {
            predictions->grad = loss_grad;
        }
    };
    
    // Set parents for gradient tracking
    loss_tensor->parents = {predictions, targets};
    
    return loss_tensor;
}

void Transformer::zero_grad() {
    encoder.zero_grad();
    decoder.zero_grad();
    linear.zero_grad();
    // Clear the embedding gradients map
    token_embedding_grads->clear();
}

void Transformer::step() {
    // Update weights using the computed gradients
    encoder.step(learning_rate);
    decoder.step(learning_rate);
    linear.step(learning_rate);
    
    // Update token embeddings using accumulated gradients
    for (auto& [token, grad] : *token_embedding_grads) {
        // Initialize embedding if it doesn't exist
        if (token_embeddings->find(token) == token_embeddings->end()) {
            (*token_embeddings)[token] = vector<float>(d_model, 0.0f);
        }
        
        // Update embedding using gradient
        for (int j = 0; j < d_model; ++j) {
            (*token_embeddings)[token][j] -= learning_rate * grad[j];
        }
    }
}


