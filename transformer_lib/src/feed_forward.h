// test, clean up includes

#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

class FeedForward {
    
    public:

        // Declare signature of constructor methods
        FeedForward(int d_model, int d_ff);
        Matrix forward(const Matrix& X) const;
    
    private:

        int d_model, d_ff;

        Matrix W_1, W_2;
        vector<float> b_1, b_2;
};

#endif // FEED_FORWARD_H