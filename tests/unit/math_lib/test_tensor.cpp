#include <math_lib/tensor.h>
#include <math_lib/matrix.h>
#include <math_lib/math_lib.h>
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;

void test_tensor_add() {
    // Test 1: Regular element-wise addition
    vector<vector<float>> arr1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    vector<vector<float>> arr2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    
    shared_ptr<Tensor> t1 = make_shared<Tensor>(arr1, true);  // Enable requires_grad
    shared_ptr<Tensor> t2 = make_shared<Tensor>(arr2, true);  // Enable requires_grad
    
    // Test forward pass
    shared_ptr<Tensor> result = math_lib::add(t1, t2);
    
    // Check dimensions
    assert(result->data.numRows() == 2);
    assert(result->data.numCols() == 2);
    
    // Check forward pass values
    assert(abs(result->data[0][0] - 6.0f) < 1e-6);  // 1 + 5
    assert(abs(result->data[0][1] - 8.0f) < 1e-6);  // 2 + 6
    assert(abs(result->data[1][0] - 10.0f) < 1e-6); // 3 + 7
    assert(abs(result->data[1][1] - 12.0f) < 1e-6); // 4 + 8
    
    // Test backward pass for regular addition
    result->grad[0][0] = 1.0f;
    result->grad[0][1] = 2.0f;
    result->grad[1][0] = 3.0f;
    result->grad[1][1] = 4.0f;
    result->backward_fn();
    
    // Check gradients for regular addition
    assert(abs(t1->grad[0][0] - 1.0f) < 1e-6);
    assert(abs(t1->grad[0][1] - 2.0f) < 1e-6);
    assert(abs(t1->grad[1][0] - 3.0f) < 1e-6);
    assert(abs(t1->grad[1][1] - 4.0f) < 1e-6);
    
    assert(abs(t2->grad[0][0] - 1.0f) < 1e-6);
    assert(abs(t2->grad[0][1] - 2.0f) < 1e-6);
    assert(abs(t2->grad[1][0] - 3.0f) < 1e-6);
    assert(abs(t2->grad[1][1] - 4.0f) < 1e-6);
    
    // Test 2: Broadcasting addition (matrix + vector)
    vector<vector<float>> arr3 = {{1.0f, 2.0f}, {3.0f, 4.0f}};  // 2x2 matrix
    vector<vector<float>> arr4 = {{5.0f}, {6.0f}};              // 2x1 vector
    
    shared_ptr<Tensor> t3 = make_shared<Tensor>(arr3, true);    // Enable requires_grad
    shared_ptr<Tensor> t4 = make_shared<Tensor>(arr4, true);    // Enable requires_grad
    
    // Test forward pass for broadcasting
    shared_ptr<Tensor> result2 = math_lib::add(t3, t4);
    
    // Check dimensions
    assert(result2->data.numRows() == 2);
    assert(result2->data.numCols() == 2);
    
    // Check forward pass values for broadcasting
    assert(abs(result2->data[0][0] - 6.0f) < 1e-6);  // 1 + 5
    assert(abs(result2->data[0][1] - 8.0f) < 1e-6);  // 2 + 6
    assert(abs(result2->data[1][0] - 8.0f) < 1e-6);  // 3 + 5
    assert(abs(result2->data[1][1] - 10.0f) < 1e-6); // 4 + 6
    
    // Test backward pass for broadcasting
    result2->grad[0][0] = 1.0f;
    result2->grad[0][1] = 2.0f;
    result2->grad[1][0] = 3.0f;
    result2->grad[1][1] = 4.0f;
    result2->backward_fn();
    
    // Check gradients for broadcasting
    // Matrix gradients should be copied directly
    assert(abs(t3->grad[0][0] - 1.0f) < 1e-6);
    assert(abs(t3->grad[0][1] - 2.0f) < 1e-6);
    assert(abs(t3->grad[1][0] - 3.0f) < 1e-6);
    assert(abs(t3->grad[1][1] - 4.0f) < 1e-6);
    
    // Vector gradients should be summed across rows
    assert(abs(t4->grad[0][0] - 4.0f) < 1e-6);  // 1 + 3 (sum of first column)
    assert(abs(t4->grad[1][0] - 6.0f) < 1e-6);  // 2 + 4 (sum of second column)
    
    cout << "PASSED test_tensor_add" << endl;
}

void test_tensor_matmul() {
    // Test matrix multiplication
    vector<vector<float>> arr1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};  // 2x2 matrix
    vector<vector<float>> arr2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};  // 2x2 matrix
    
    shared_ptr<Tensor> t1 = make_shared<Tensor>(arr1, true);  // Enable requires_grad
    shared_ptr<Tensor> t2 = make_shared<Tensor>(arr2, true);  // Enable requires_grad
    
    // Test forward pass
    shared_ptr<Tensor> result = math_lib::matmul(t1, t2);
    
    // Check dimensions
    assert(result->data.numRows() == 2);
    assert(result->data.numCols() == 2);
    
    // Check forward pass values
    // For 2x2 matrix multiplication:
    // {{1,2}, {3,4}} * {{5,6}, {7,8}} = {{1*5+2*7, 1*6+2*8}, {3*5+4*7, 3*6+4*8}}
    // = {{19, 22}, {43, 50}}
    assert(abs(result->data[0][0] - 19.0f) < 1e-6);  // 1*5 + 2*7
    assert(abs(result->data[0][1] - 22.0f) < 1e-6);  // 1*6 + 2*8
    assert(abs(result->data[1][0] - 43.0f) < 1e-6);  // 3*5 + 4*7
    assert(abs(result->data[1][1] - 50.0f) < 1e-6);  // 3*6 + 4*8
    
    // Test backward pass
    result->grad[0][0] = 1.0f;
    result->grad[0][1] = 2.0f;
    result->grad[1][0] = 3.0f;
    result->grad[1][1] = 4.0f;
    result->backward_fn();
    
    // Check gradients
    // For t1 (first matrix):
    // grad = result_grad * t2.T()
    // {{1,2}, {3,4}} * {{5,7}, {6,8}} = {{1*5+2*6, 1*7+2*8}, {3*5+4*6, 3*7+4*8}}
    // = {{17, 23}, {39, 53}}
    assert(abs(t1->grad[0][0] - 17.0f) < 1e-6);  // 1*5 + 2*6
    assert(abs(t1->grad[0][1] - 23.0f) < 1e-6);  // 1*7 + 2*8
    assert(abs(t1->grad[1][0] - 39.0f) < 1e-6);  // 3*5 + 4*6
    assert(abs(t1->grad[1][1] - 53.0f) < 1e-6);  // 3*7 + 4*8
    
    // For t2 (second matrix):
    // grad = t1.T() * result_grad
    // {{1,3}, {2,4}} * {{1,2}, {3,4}} = {{1*1+3*3, 1*2+3*4}, {2*1+4*3, 2*2+4*4}}
    // = {{10, 14}, {14, 20}}
    assert(abs(t2->grad[0][0] - 10.0f) < 1e-6);  // 1*1 + 3*3
    assert(abs(t2->grad[0][1] - 14.0f) < 1e-6);  // 1*2 + 3*4
    assert(abs(t2->grad[1][0] - 14.0f) < 1e-6);  // 2*1 + 4*3
    assert(abs(t2->grad[1][1] - 20.0f) < 1e-6);  // 2*2 + 4*4
    
    cout << "PASSED test_tensor_matmul" << endl;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    test_tensor_add();
    test_tensor_matmul();
    
    cout << "All tensor tests passed!" << endl;
    return 0;
}
