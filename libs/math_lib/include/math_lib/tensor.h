#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_set>

#include <math_lib/matrix.h>

using namespace std;

class Tensor : public enable_shared_from_this<Tensor> {
    
    public:

        Tensor();

        Tensor(const vector<int>& shape, bool requires_grad=false);

        Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad=false);

        int size();
        void backward();

        bool requires_grad;
        vector<int> shape;
        vector<int> strides;
        int* data;
        shared_ptr<Tensor> grad;
        vector<shared_ptr<Tensor>> parents;
        function<void()> backward_fn;

        // // Default constructor
        // Tensor();
        
        // // Size constructor
        // Tensor(size_t rows, size_t cols, bool requires_grad=false);
        
        // // 2D vector constructor
        // Tensor(const vector<vector<float>> arr, bool requires_grad=false);
        
        // // 1D vector constructor
        // Tensor(const vector<float> arr, bool requires_grad=false);
        
        // // Existing Matrix constructor
        // Tensor(const Matrix& data, bool requires_grad=false);
        
        // // Backward pass
        // void backward();
        
        // bool requires_grad;
        // Matrix data;
        // Matrix grad;
        // vector<shared_ptr<Tensor>> parents;
        // function<void()> backward_fn;
};

#endif // TENSOR_H