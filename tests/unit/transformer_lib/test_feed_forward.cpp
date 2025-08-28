#include <transformer_lib/feed_forward.h>
#include <math_lib/math_lib.h>
#include <iostream>
#include <cassert>

using namespace std;

// void test_feed_forward() {
//     int d_model = 512;
//     int d_ff = 2048;
//     FeedForward ff(d_model, d_ff);
    
//     // Create test input
//     Matrix input(2, d_model);
//     for (int i = 0; i < input.numRows(); ++i) {
//         for (int j = 0; j < input.numCols(); ++j) {
//             input[i][j] = 0.1f;
//         }
//     }
    
//     // Perform forward pass
//     Matrix output = ff.forward(input);
    
//     // Check output dimensions
//     assert(output.numRows() == input.numRows());
//     assert(output.numCols() == d_model);
    
//     // Check that output contains valid values
//     for (int i = 0; i < output.numRows(); ++i) {
//         for (int j = 0; j < output.numCols(); ++j) {
//             assert(!std::isnan(output[i][j]));
//             assert(!std::isinf(output[i][j]));
//         }
//     }
    
//     cout << "PASSED test_feed_forward" << endl;
// }

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    // test_feed_forward();
    
    return 0;
} 