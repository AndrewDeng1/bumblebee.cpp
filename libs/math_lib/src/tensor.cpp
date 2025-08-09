#include <math_lib/tensor.h>

Tensor::Tensor() : data(nullptr), grad(nullptr), parents(), backward_fn() {}

Tensor::Tensor(const vector<int>& shape, bool requires_grad) : shape(shape), requires_grad(requires_grad) {
    
    this->requires_grad = requires_grad;
    this->shape = shape;
    
    this->strides = vector<int>(shape.size());
    this->strides[shape.size()-1] = 1;

    for(int i=shape.size()-2; i>=0; i--){
        this->strides[i] = this->strides[i+1] * shape[i+1];
    }

    this->data = new int[size()];

    if(requires_grad) {
        this->grad = make_shared<Tensor>(shape, false);
    } else {
        this->grad = nullptr;
    }

    this->parents = vector<shared_ptr<Tensor>>();
    this->backward_fn = [](){};
}

Tensor::Tensor(const vector<int>& shape, const vector<float>& data, bool requires_grad) : Tensor(shape, requires_grad) {
    
    for(int i=0; i<size(); i++){
        this->data[i] = data[i];
    }
}

int Tensor::size(){
    int cnt=1;
    for(int i=0; i<this->shape.size(); i++){
        cnt *= this->shape[i];
    }
    return cnt;
}

void Tensor::backward() {
    // Topological sort using Kahn's algorithm
    unordered_map<shared_ptr<Tensor>, int> in_degree;
    queue<shared_ptr<Tensor>> q;
    unordered_set<shared_ptr<Tensor>> visited;
    
    // First, traverse the entire graph to count in-degrees for all tensors
    function<void(shared_ptr<Tensor>)> count_in_degrees = [&](shared_ptr<Tensor> tensor) {
        if (visited.count(tensor)) return;
        visited.insert(tensor);
        
        // Count incoming edges for this tensor's parents
        for (const auto& parent : tensor->parents) {
            in_degree[parent]++;
            count_in_degrees(parent);  // Recursively process parents
        }
    };
    
    // Start counting from this tensor
    count_in_degrees(shared_from_this());
    
    // Add tensors with no incoming edges to queue
    for (const auto& [tensor, degree] : in_degree) {
        if (degree == 0) {
            q.push(tensor);
        }
    }
    
    // Process tensors in topological order
    while (!q.empty()) {
        auto tensor = q.front();
        q.pop();
        
        // Call backward function if it exists
        if (tensor->backward_fn) {
            tensor->backward_fn();
        }
        
        // Update in-degree for children
        for (const auto& child : tensor->parents) {
            in_degree[child]--;
            if (in_degree[child] == 0) {
                q.push(child);
            }
        }
    }
}


// // Default constructor
// Tensor::Tensor(): data(Matrix()), requires_grad(false) {}

// // Size constructor
// Tensor::Tensor(size_t rows, size_t cols, bool requires_grad): 
//     data(Matrix(rows, cols)), requires_grad(requires_grad) {
//     if(requires_grad) {
//         grad = Matrix(rows, cols);
//         parents = vector<shared_ptr<Tensor>>();
//         backward_fn = [](){};
//     }
// }

// // 2D vector constructor
// Tensor::Tensor(const vector<vector<float>> arr, bool requires_grad): 
//     data(Matrix(arr)), requires_grad(requires_grad) {
//     if(requires_grad) {
//         grad = Matrix(data.numRows(), data.numCols());
//         parents = vector<shared_ptr<Tensor>>();
//         backward_fn = [](){};
//     }
// }

// // 1D vector constructor
// Tensor::Tensor(const vector<float> arr, bool requires_grad): 
//     data(Matrix(arr)), requires_grad(requires_grad) {
//     if(requires_grad) {
//         grad = Matrix(data.numRows(), data.numCols());
//         parents = vector<shared_ptr<Tensor>>();
//         backward_fn = [](){};
//     }
// }

// // Existing Matrix constructor
// Tensor::Tensor(const Matrix& data, bool requires_grad): data(data), requires_grad(requires_grad) {
//     if(requires_grad) {
//         grad = Matrix(data.numRows(), data.numCols());
//         parents = vector<shared_ptr<Tensor>>();
//         backward_fn = [](){};
//     }
// }

// void Tensor::backward() {
//     // Topological sort using Kahn's algorithm
//     unordered_map<shared_ptr<Tensor>, int> in_degree;
//     queue<shared_ptr<Tensor>> q;
//     unordered_set<shared_ptr<Tensor>> visited;
    
//     // First, traverse the entire graph to count in-degrees for all tensors
//     function<void(shared_ptr<Tensor>)> count_in_degrees = [&](shared_ptr<Tensor> tensor) {
//         if (visited.count(tensor)) return;
//         visited.insert(tensor);
        
//         // Count incoming edges for this tensor's parents
//         for (const auto& parent : tensor->parents) {
//             in_degree[parent]++;
//             count_in_degrees(parent);  // Recursively process parents
//         }
//     };
    
//     // Start counting from this tensor
//     count_in_degrees(shared_from_this());
    
//     // Add tensors with no incoming edges to queue
//     for (const auto& [tensor, degree] : in_degree) {
//         if (degree == 0) {
//             q.push(tensor);
//         }
//     }
    
//     // Process tensors in topological order
//     while (!q.empty()) {
//         auto tensor = q.front();
//         q.pop();
        
//         // Call backward function if it exists
//         if (tensor->backward_fn) {
//             tensor->backward_fn();
//         }
        
//         // Update in-degree for children
//         for (const auto& child : tensor->parents) {
//             in_degree[child]--;
//             if (in_degree[child] == 0) {
//                 q.push(child);
//             }
//         }
//     }
// }