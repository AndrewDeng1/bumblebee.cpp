// test, clean up include

#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <vector>
#include <iostream>
#include <cassert>
#include "../../math_lib/src/math_lib.h"
#include "head.h"

using namespace std;

class MultiHeadAttention {
    
    public:

        // Declare signature of constructor methods
        MultiHeadAttention(int d_model, int h, int d_k, int d_v, bool masked=false);
        Matrix forward(Matrix& Q, Matrix& K, Matrix& V) const;
    
    private:

        int d_model;
        int h;
        bool masked;

        vector<Head> heads;
        Matrix W_O;
};

#endif // MULTI_HEAD_ATTENTION_H