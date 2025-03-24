// test, clean up include

#ifndef HEAD_H
#define HEAD_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

class Head {
    
    public:

        // Declare signature of constructor methods
        Head(int d_model, int d_k, int d_v, bool masked);
        Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V) const;
    
    private:

        int d_model, d_k, d_v;
        bool masked;
        Matrix W_Q, W_K, W_V;
};

#endif // HEAD_H