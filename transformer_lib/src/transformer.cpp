#include "transformer.h"

Transformer::Transformer(int d_model, int V, int d_ff, int h, int d_k, int d_v, int N) {
    this->d_model=d_model;
    this->V=V;
    this->d_ff=d_ff;
    this->h=h;
    this->d_k=d_k;
    this->d_v=d_v;
    this->N=N;

    // Initialize embeddings

    this->encoder=Encoder(d_model, d_ff, h, d_k, d_v, N);
    this->decoder=Decoder(d_model, d_ff, h, d_k, d_v, N);
    this->linear=Linear(d_model, V);
}

Matrix Transformer::forward(const vector<string>& inputs, const vector<string>& outputs) const {

    Matrix input_embeddings = embed(inputs);
    Matrix output_embeddings = embed(outputs);  // Shifted right

    Matrix X_input = positional_encoder(input_embeddings, d_model);
    Matrix X_output = positional_encoder(output_embeddings, d_model);

    Matrix encoder_out = encoder.forward(X_input);
    Matrix decoder_out = decoder.forward(X_output, encoder_out);

    return softmax(linear.forward(decoder_out));
}