#include <transformer_lib/multi_head_attention.h>
#include <math_lib/math_lib.h>
#include <iostream>
#include <cassert>

using namespace std;

void test_forward_pass() {
    int d_model = 512;
    int h = 8;      // number of heads
    int d_k = 64;   // dimension of key
    int d_v = 64;   // dimension of value
    bool masked = false;
    MultiHeadAttention mha(d_model, h, d_k, d_v, masked);
    
    // Create test matrices
    Matrix Q(2, d_model);  // batch_size=2, d_model=512
    Matrix K(3, d_model);  // seq_len=3, d_model=512
    Matrix V(3, d_model);  // seq_len=3, d_model=512
    
    // Initialize with test values
    for (int i = 0; i < Q.numRows(); ++i) {
        for (int j = 0; j < Q.numCols(); ++j) {
            Q[i][j] = 0.1f;
        }
    }
    for (int i = 0; i < K.numRows(); ++i) {
        for (int j = 0; j < K.numCols(); ++j) {
            K[i][j] = 0.2f;
            V[i][j] = 0.3f;
        }
    }
    
    // Perform forward pass
    Matrix output = mha.forward(Q, K, V);
    
    // Check output dimensions
    assert(output.numRows() == Q.numRows());  // Should match batch size
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

void test_masked_attention() {
    int d_model = 512;
    int h = 8;      // number of heads
    int d_k = 64;   // dimension of key
    int d_v = 64;   // dimension of value
    bool masked = true;
    MultiHeadAttention mha(d_model, h, d_k, d_v, masked);
    
    // Create test matrices
    Matrix Q(2, d_model);
    Matrix K(3, d_model);
    Matrix V(3, d_model);
    
    // Initialize with test values
    for (int i = 0; i < Q.numRows(); ++i) {
        for (int j = 0; j < Q.numCols(); ++j) {
            Q[i][j] = 0.1f;
        }
    }
    for (int i = 0; i < K.numRows(); ++i) {
        for (int j = 0; j < K.numCols(); ++j) {
            K[i][j] = 0.2f;
            V[i][j] = 0.3f;
        }
    }
    
    // Perform forward pass
    Matrix output = mha.forward(Q, K, V);
    
    // Check output dimensions
    assert(output.numRows() == Q.numRows());
    assert(output.numCols() == d_model);
    
    // Check that output contains valid values
    for (int i = 0; i < output.numRows(); ++i) {
        for (int j = 0; j < output.numCols(); ++j) {
            assert(!std::isnan(output[i][j]));
            assert(!std::isinf(output[i][j]));
        }
    }
    
    cout << "PASSED test_masked_attention" << endl;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    test_forward_pass();
    test_masked_attention();
    
    return 0;
} 