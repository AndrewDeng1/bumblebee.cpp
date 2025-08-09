#include "tensor.h"
#include <iostream>

int main() {

    // // Create a 2x3 tensor
    vector<int> shape = {2, 4, 3};
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    auto t1 = make_shared<Tensor>(shape, data, true);
    auto t2 = make_shared<Tensor>(vector<int>{1}, vector<float>{2.0}, true);
    auto t3 = t2/t1;
    t3->print();
    t3->grad = make_shared<Tensor>(vector<int>{1}, vector<float>{1.0}, true);
    t3->backward();
    t1->grad->print();
    t2->grad->print();

    auto t4 = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    auto t5 = make_shared<Tensor>(vector<int>{2, 1}, vector<float>{1.0f, 2.0f}, true);
    auto t6 = t4/t5;
    t6->print();
    t6->grad = make_shared<Tensor>(vector<int>{2, 2, 3}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, true);
    t6->backward();
    t4->grad->print();
    t5->grad->print();
}

// Tensor(2, 4, 3):
// [[2 , 1 , 0.666667 ], [0.5 , 0.4 , 0.333333 ], [0.285714 , 0.25 , 0.222222 ], [0.2 , 0.181818 , 0.166667 ]], [[0.153846 , 0.142857 , 0.133333 ], [0.125 , 0.117647 , 0.111111 ], [0.105263 , 0.1 , 0.0952381 ], [0.0909091 , 0.0869565 , 0.0833333 ]]
// Tensor(2, 4, 3):
// [[-2, -0.5, -0.222222], [-0.125, -0.08, -0.0555556], [-0.0408163, -0.03125, -0.0246914], [-0.02, -0.0165289, -0.0138889]], [[-0.0118343, -0.0102041, -0.00888889], [-0.0078125, -0.00692042, -0.00617284], [-0.00554017, -0.005, -0.00453515], [-0.00413223, -0.00378072, -0.00347222]]
// Tensor(1):
// 3.77596
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [0.5 , 1 , 1.5 ]], [[4 , 5 , 6 ], [2 , 2.5 , 3 ]]
// Tensor(2, 1, 3):
// [[1.5 , 1.5 , 1.5 ]], [[1.5 , 1.5 , 1.5 ]]
// Tensor(2, 1):
// [-21], [-5.25]
// [-21], [-5.25]