// test, clean up includes

#ifndef ENCODER_H
#define ENCODER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <math_lib/math_lib.h>
#include <transformer_lib/encoder_layer.h>

using namespace std;

class Encoder {
    
    public:

        // Declare signature of constructor methods
        Encoder(int d_model, int d_ff, int h, int d_k, int d_v, int N);
        Matrix forward(const Matrix& X) const;

    private:

        int d_model;
        int d_ff;
        int h;
        int d_k;
        int d_v;
        int N;
        
        vector<EncoderLayer> encoder_layers;
};

#endif // ENCODER_H