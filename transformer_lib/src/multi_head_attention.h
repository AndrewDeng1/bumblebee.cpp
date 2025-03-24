// test, clean up include

#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

class MultiHeadAttention {
    
    public:

        // Declare signature of constructor methods
        MultiHeadAttention(int d_model, int h, bool masked);
        Matrix forward(const Matrix& X) const;
    
    private:

        int d_model;
        int h;
        bool masked;

        vector<Head> heads;
        Matrix W_O;
};

#endif // MULTI_HEAD_ATTENTION_H