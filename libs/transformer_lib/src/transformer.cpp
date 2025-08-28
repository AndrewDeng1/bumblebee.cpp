#include <transformer_lib/transformer.h>
#include <tensor_lib/tensor.h>
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
      linear(d_model, V),
      token_embeddings(make_shared<Tensor>(Tensor({V, d_model}, true))) {  // Initialize with requires_grad=true
    // Initialize token embeddings with Xavier uniform initialization
    tensor_lib::xavier_uniform_initialization(token_embeddings);
}

shared_ptr<Tensor> Transformer::forward(
    const vector<vector<int>>& input_tokens,
    const vector<vector<int>>& output_tokens
) const {   
    // Get embeddings for input and output sequences
    auto input_embeddings = tensor_lib::embed(input_tokens, token_embeddings, d_model);
    auto output_embeddings = tensor_lib::embed(output_tokens, token_embeddings, d_model);
    
    // Add positional encoding
    input_embeddings = tensor_lib::positional_encoder(input_embeddings, d_model);
    output_embeddings = tensor_lib::positional_encoder(output_embeddings, d_model);
    
    // Forward pass through encoder
    auto encoder_out = encoder.forward(input_embeddings);
    
    // Forward pass through decoder
    auto decoder_out = decoder.forward(output_embeddings, encoder_out);
    
    // Final linear layer and softmax
    return tensor_lib::softmax(linear.forward(decoder_out));
}

// // targets.requires_grad = false, but doesn't rly matter
// shared_ptr<Tensor> Transformer::cross_entropy_loss(
//     const shared_ptr<Tensor>& predictions,
//     const shared_ptr<Tensor>& targets
// ) const {
//     int seq_len = predictions->data.numRows();
//     int vocab_size = predictions->data.numCols();
    
//     // Create a new tensor for the loss
//     auto loss_tensor = make_shared<Tensor>(Matrix(1, 1), true);  // Single scalar value
//     float total_loss = 0.0f;
    
//     // Compute cross entropy loss
//     for (int i = 0; i < seq_len; i++) {
//         for (int j = 0; j < vocab_size; j++) {
//             if (targets->data[i][j] > 0) {  // Only compute loss for the target token
//                 float pred = predictions->data[i][j];
//                 float target = targets->data[i][j];
//                 total_loss -= target * log(pred + 1e-10f);  // Add small epsilon for numerical stability
//             }
//         }
//     }
    
//     // Set the loss value
//     loss_tensor->data[0][0] = total_loss;
    
//     // Set up the backward function
//     loss_tensor->backward_fn = [predictions, targets]() {
//         int seq_len = predictions->data.numRows();
//         int vocab_size = predictions->data.numCols();
//         Matrix loss_grad(seq_len, vocab_size);
        
//         // Compute gradient: prediction - target
//         for (int i = 0; i < seq_len; i++) {
//             for (int j = 0; j < vocab_size; j++) {
//                 if (targets->data[i][j] > 0) {
//                     loss_grad[i][j] = predictions->data[i][j] - targets->data[i][j];
//                 }
//             }
//         }
        
//         // Set gradients for parent tensors
//         if (predictions->requires_grad) {
//             predictions->grad = loss_grad;
//         }
//     };
    
//     // Set parents for gradient tracking
//     loss_tensor->parents = {predictions, targets};
    
//     return loss_tensor;
// }

void Transformer::zero_grad() {
    encoder.zero_grad();
    decoder.zero_grad();
    linear.zero_grad();
    token_embeddings->zero_grad();
}

void Transformer::step() {
    // Update weights using the computed gradients
    encoder.step(learning_rate);
    decoder.step(learning_rate);
    linear.step(learning_rate);
    token_embeddings->step(learning_rate);
}


