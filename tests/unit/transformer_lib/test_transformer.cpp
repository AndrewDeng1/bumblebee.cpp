#include <transformer_lib/transformer.h>
#include <math_lib/math_lib.h>
#include <iostream>
#include <cassert>

using namespace std;

// void test_forward_pass() {
//     int d_model = 512;
//     int V = 1000;     // vocabulary size
//     int d_ff = 2048;  // feed-forward dimension
//     int h = 8;        // number of heads
//     int d_k = 64;     // dimension of key
//     int d_v = 64;     // dimension of value
//     int N = 6;        // number of layers
//     Transformer transformer(d_model, V, d_ff, h, d_k, d_v, N);
    
//     // Create test input and output sequences
//     vector<string> inputs = {"hello", "world"};
//     vector<string> outputs = {"<s>", "bonjour", "le", "monde"};
    
//     // Create a dummy input matrix with correct dimensions
//     Matrix input_matrix(inputs.size(), d_model);
//     for (int i = 0; i < input_matrix.numRows(); ++i) {
//         for (int j = 0; j < input_matrix.numCols(); ++j) {
//             input_matrix[i][j] = 0.1f;  // Fill with dummy values for now
//         }
//     }
    
//     // Perform forward pass
//     Matrix output = transformer.forward(inputs, outputs);
    
//     // Check output dimensions
//     assert(output.numRows() == outputs.size());  // Should match output sequence length
//     assert(output.numCols() == V);               // Should match vocabulary size
    
//     // Check that output contains valid values
//     for (int i = 0; i < output.numRows(); ++i) {
//         for (int j = 0; j < output.numCols(); ++j) {
//             assert(!std::isnan(output[i][j]));
//             assert(!std::isinf(output[i][j]));
//             assert(output[i][j] >= 0.0f);        // Softmax output should be non-negative
//             assert(output[i][j] <= 1.0f);        // Softmax output should be <= 1
//         }
//     }
    
//     cout << "PASSED test_forward_pass" << endl;
// }

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    // test_forward_pass();
    
    return 0;
} 