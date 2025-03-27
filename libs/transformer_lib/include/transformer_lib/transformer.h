// test, clean up includes

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <math_lib/math_lib.h>
#include <transformer_lib/encoder.h>
#include <transformer_lib/decoder.h>
#include <transformer_lib/linear.h>

using namespace std;

class Transformer {
    
    public:

        // Declare signature of constructor methods
        Transformer(int d_model, int V, int d_ff, int h, int d_k, int d_v, int N);
        Matrix forward(const vector<string>& inputs, const vector<string>& outputs) const;

    private:

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

        Matrix embed(const vector<string>& inputs) const;
};

#endif // TRANSFORMER_H
