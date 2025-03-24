// test, clean up includes

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

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