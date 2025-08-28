#include "tensor.h"
#include <iostream>

// reminder u should test like A, then B = A.T(), then C = A*B, then backward, see if correct A and B grad

int main() {
    
    // // Create a 2x3 tensor
    vector<int> shape = {2, 3, 4};
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    auto t1 = make_shared<Tensor>(shape, data, true);
    auto t2 = make_shared<Tensor>(vector<int>{4, 2}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, true);
    auto t3 = matmul(t1, t2);
    t1->print();
    t2->print();
    t3->print();
    t3->grad = make_shared<Tensor>(vector<int>{2, 3, 2}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, true);
    t3->backward();
    t1->grad->print();
    t2->grad->print();

    auto t4 = make_shared<Tensor>(vector<int>{1, 5}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
    auto t5 = make_shared<Tensor>(vector<int>{5, 1}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
    auto t6 = matmul(t4, t5);
    t6->print();
    t6->grad = make_shared<Tensor>(vector<int>{1, 1}, vector<float>{1.0f}, true);
    t6->backward();
    t4->grad->print();
    t5->grad->print();

    // auto t7 = make_shared<Tensor>(vector<int>{1, 5}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
    // auto t8 = make_shared<Tensor>(vector<int>{4, 1}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, true);
    // auto t9 = matmul(t7, t8);

    // auto t10 = make_shared<Tensor>(vector<int>{1}, vector<float>{1.0f}, true);
    // auto t11 = matmul(t10, t9);
}



// Tensor(2, 3, 4):
// [[1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ], [9 , 10 , 11 , 12 ]], [[13 , 14 , 15 , 16 ], [17 , 18 , 19 , 20 ], [21 , 22 , 23 , 24 ]]
// Tensor(4, 2):
// [1 , 2 ], [3 , 4 ], [5 , 6 ], [7 , 8 ]
// Tensor(2, 3, 2):
// [[50 , 60 ], [114 , 140 ], [178 , 220 ]], [[242 , 300 ], [306 , 380 ], [370 , 460 ]]
// Tensor(2, 3, 4):
// [[5 , 11 , 17 , 23 ], [11 , 25 , 39 , 53 ], [17 , 39 , 61 , 83 ]], [[23 , 53 , 83 , 113 ], [29 , 67 , 105 , 143 ], [35 , 81 , 127 , 173 ]]
// Tensor(4, 2):
// [536 , 602 ], [572 , 644 ], [608 , 686 ], [644 , 728 ]
// Tensor(1, 1):
// [55 ]
// Tensor(1, 5):
// [1 , 2 , 3 , 4 , 5 ]
// Tensor(5, 1):
// [1 ], [2 ], [3 ], [4 ], [5 ]