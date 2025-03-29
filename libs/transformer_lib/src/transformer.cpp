#include <transformer_lib/transformer.h>

Transformer::Transformer(int d_model, int V, int d_ff, int h, int d_k, int d_v, int N)
    : d_model(d_model),
      V(V),
      d_ff(d_ff),
      h(h),
      d_k(d_k),
      d_v(d_v),
      N(N),
      encoder(d_model, d_ff, h, d_k, d_v, N),
      decoder(d_model, d_ff, h, d_k, d_v, N),
      linear(d_model, V) {
    // Empty body
}

Matrix Transformer::forward(const vector<string>& inputs, const vector<string>& outputs) const {
    // Create input matrix with correct dimensions
    Matrix input_matrix(inputs.size(), d_model);
    for (int i = 0; i < input_matrix.numRows(); ++i) {
        for (int j = 0; j < input_matrix.numCols(); ++j) {
            input_matrix[i][j] = 0.1f;  // Fill with dummy values for now
        }
    }

    // Matrix input_embeddings = embed(inputs);
    // Matrix output_embeddings = embed(outputs);  // Shifted right

    // Matrix X_input = math_lib::positional_encoder(input_embeddings, d_model);
    // Matrix X_output = math_lib::positional_encoder(output_embeddings, d_model);

    // Matrix encoder_out = encoder.forward(X_input);
    // Matrix decoder_out = decoder.forward(X_output, encoder_out);

    // return math_lib::softmax(linear.forward(decoder_out));

    // For now, return a dummy matrix
    Matrix dummy(outputs.size(), V);
    for (int i = 0; i < dummy.numRows(); ++i) {
        for (int j = 0; j < dummy.numCols(); ++j) {
            dummy[i][j] = 0.1f;
        }
    }
    return dummy;
}