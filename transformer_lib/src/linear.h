// test, clean up includes

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include <iostream>
#include <cassert>
#include "../../math_lib/src/math_lib.h"

using namespace std;

class Linear {
    
    public:

        // Declare signature of constructor methods
        Linear(int d_model, int V);
        Matrix forward(const Matrix& X) const;
    
    private:

        int d_model;
        int V;

        Matrix W;
        vector<float>b;
};

#endif // LINEAR_LAYER_H