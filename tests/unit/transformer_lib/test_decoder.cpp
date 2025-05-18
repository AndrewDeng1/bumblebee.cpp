// #include<bits/stdc++.h>
#define ll long long
using namespace std;

#include <math_lib/matrix.h>
// #include "math_lib/src/matrix.h"
#include <transformer_lib/decoder.h>
#include <math_lib/math_lib.h>
#include <iostream>
#include <cassert>

// void test_init(){
//     Matrix m=Matrix();
//     assert(m.numRows()==0&&m.numCols()==0);
    
//     Matrix a=Matrix(5, 10);
//     assert(a.numRows()==5&&a.numCols()==10);

//     cout<<"PASSED test_init"<<endl;
// }

// void test_operators(){
    
//     // vector<float>row1={1, 2};
//     // vector<float>row2={4, 6};
//     // vector<float>row3={8, 10};

//     // vector<vector<float>>arr1={row1, row2, row3};

//     // Matrix m=Matrix(arr1);
//     // cout<<"PASSED test_operators"<<endl;
// }

// void test_forward_pass() {
//     int d_model = 512;
//     int d_ff = 2048;  // feed-forward dimension
//     int h = 8;        // number of heads
//     int d_k = 64;     // dimension of key
//     int d_v = 64;     // dimension of value
//     int N = 6;        // number of decoder layers
//     Decoder decoder(d_model, d_ff, h, d_k, d_v, N);
    
//     // Create input matrices
//     Matrix X(2, d_model);        // decoder input (batch_size=2, d_model=512)
//     Matrix encoder_out(2, d_model);  // encoder output
    
//     // Initialize with test values
//     for (int i = 0; i < X.numRows(); ++i) {
//         for (int j = 0; j < X.numCols(); ++j) {
//             X[i][j] = 0.1f;
//             encoder_out[i][j] = 0.2f;
//         }
//     }
    
//     // Perform forward pass
//     Matrix output = decoder.forward(X, encoder_out);
    
//     // Check output dimensions
//     assert(output.numRows() == X.numRows());  // Should match batch size
//     assert(output.numCols() == d_model);      // Should match d_model
    
//     // Check that output contains valid values
//     for (int i = 0; i < output.numRows(); ++i) {
//         for (int j = 0; j < output.numCols(); ++j) {
//             assert(!std::isnan(output[i][j]));
//             assert(!std::isinf(output[i][j]));
//         }
//     }
    
//     cout << "PASSED test_forward_pass" << endl;
// }

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);

    // test_init();
    // test_operators();
    // test_forward_pass();
}