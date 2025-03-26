// test, clean up includes

#ifndef DECODER_H
#define DECODER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <math_lib/math_lib.h>
#include <transformer_lib/decoder_layer.h>

using namespace std;

class Decoder {
    
    public:

        // Declare signature of constructor methods
        Decoder(int d_model, int d_ff, int h, int d_k, int d_v, int N);
        Matrix forward(const Matrix& X, const Matrix& encoder_out) const;

    private:

        int d_model;
        int d_ff;
        int h;
        int d_k;
        int d_v;
        int N;
        
        vector<DecoderLayer> decoder_layers;
};

#endif // DECODER_H