#include <transformer_lib/feed_forward.h>
#include <math_lib/math_lib.h>
#include <iostream>
#include <cassert>

using namespace std;

void test_forward_pass() {
    int d_model = 512;
    int d_ff = 2048;  // feed-forward dimension
    FeedForward ff(d_model, d_ff);
    
    // Create input matrix (batch_size=2, d_model=512)
    Matrix x(2, d_model);
    
    // Initialize with test values
    for (int i = 0; i < x.numRows(); ++i) {
        for (int j = 0; j < x.numCols(); ++j) {
            x[i][j] = 0.1f;
        }
    }
    
    // Perform forward pass
    Matrix output = ff.forward(x);
    
    // Check output dimensions
    assert(output.numRows() == x.numRows());  // Should match batch size
    assert(output.numCols() == d_model);      // Should match d_model
    
    // Check that output contains valid values
    for (int i = 0; i < output.numRows(); ++i) {
        for (int j = 0; j < output.numCols(); ++j) {
            assert(!std::isnan(output[i][j]));
            assert(!std::isinf(output[i][j]));
        }
    }
    
    cout << "PASSED test_forward_pass" << endl;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    test_forward_pass();
    
    return 0;
} 