// it works 😝

#include "tensor.h"
#include <iostream>

int main() {



    // // Create a 2x3 tensor
    vector<int> shape = {2, 4, 3};
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    auto t1 = make_shared<Tensor>(shape, data, true);
    auto t2 = t1->reshape({2, 3, 4});
    auto t3 = t2->reshape({2, 12});
    auto t4 = t3->sum(0, true);
    auto t5 = t3->sum(1, true);

    int sm1=0;
    int sm2=0;
    for(int i=0; i<data.size()/2; i++){
        sm1+=data[i];
        sm2+=data[i+data.size()/2];
    }
    // cout << sm1 << endl;
    // cout << sm2 << endl;

    auto t6 = t1->sum(1, true);
    auto t7 = t1->reduce_to_shape({2, 1, 1});
    auto t8 = t1 + t7;

    auto a1 = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 100.0f, 1000.0f, 750.0f, 500.0f, 5000.0f}, true);
    a1->print();
    t1->print();
    auto a2 = a1+t1;
    // printf("a2 shape: %d, %d, %d\n", a2->shape[0], a2->shape[1], a2->shape[2]);
    // cout << t1->at({0, 0, 0}) << endl;
    // cout << t1->at({1, 0, 0}) << endl;
    // cout << t1->at({0, 1, 0}) << endl;
    // cout << t1->at({1, 1, 0}) << endl;
    // cout << t1->at({0, 2, 0}) << endl;
    // cout << t1->at({1, 2, 0}) << endl;
    // cout << t1->at({0, 3, 0}) << endl;
    // cout << t1->at({1, 3, 0}) << endl;
    a2->print();
    auto a3 = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    auto a4 = make_shared<Tensor>(vector<int>{3, 1}, vector<float>{100.0f, 200.0f, 300.0f}, true);
    auto a5 = a3+a4;
    a5->print();
    a5->grad = make_shared<Tensor>(vector<int>{2, 3, 3}, vector<float>{1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f, 6000.0f, 7000.0f, 8000.0f, 9000.0f, 10000.0f, 11000.0f, 12000.0f, 13000.0f, 14000.0f, 15000.0f, 16000.0f, 17000.0f, 18000.0f}, true);
    a5->grad->print();
    a5->backward();
    a3->grad->print();
    a4->grad->print();

    // grad
    // 1000 2000 3000   10000 11000 12000
    // 4000 5000 6000   13000 14000 15000
    // 7000 8000 9000   16000 17000 18000

    // a3->grad
    // 5000+4000+7000 2000+5000+8000 3000+6000+9000    10000+13000+16000 11000+14000+17000 12000+15000+18000
    // = 16000 15000 18000    33000 42000 45000

    // a4->grad
    // (1000+10000) + (2000+11000) + (3000+12000)
    // (4000+13000) + (5000+14000) + (6000+15000)
    // (7000+16000) + (8000+17000) + (9000+18000)
    // = 11000+13000+15000
    // = 17000+19000+21000
    // = 23000+25000+27000
    // = 39000
    // = 57000
    // = 75000

    auto a6 = make_shared<Tensor>(vector<int>{1, 1, 1, 1}, vector<float>{1.0f}, true);
    auto a7 = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);

    auto a8 = a6+a7;
    a8->print();
    a8->grad = make_shared<Tensor>(vector<int>{1, 1, 2, 3}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, true);
    a8->backward();
    a6->grad->print();
    a7->grad->print();
} 





// Tensor(2, 1, 3):
// [[1 , 100 , 1000 ]], [[750 , 500 , 5000 ]]
// Tensor(2, 4, 3):
// [[1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ], [10 , 11 , 12 ]], [[13 , 14 , 15 ], [16 , 17 , 18 ], [19 , 20 , 21 ], [22 , 23 , 24 ]]
// Tensor(2, 4, 3):
// [[2 , 102 , 1003 ], [5 , 105 , 1006 ], [8 , 108 , 1009 ], [11 , 111 , 1012 ]], [[763 , 514 , 5015 ], [766 , 517 , 5018 ], [769 , 520 , 5021 ], [772 , 523 , 5024 ]]
// Tensor(2, 3, 3):
// [[101 , 102 , 103 ], [201 , 202 , 203 ], [301 , 302 , 303 ]], [[104 , 105 , 106 ], [204 , 205 , 206 ], [304 , 305 , 306 ]]
// Tensor(2, 3, 3):
// [[1000 , 2000 , 3000 ], [4000 , 5000 , 6000 ], [7000 , 8000 , 9000 ]], [[10000 , 11000 , 12000 ], [13000 , 14000 , 15000 ], [16000 , 17000 , 18000 ]]
// q size: 1
// Backward +
// Backward broadcast
// Backward broadcast
// Tensor(2, 1, 3):
// [[12000 , 15000 , 18000 ]], [[39000 , 42000 , 45000 ]]
// Tensor(3, 1):
// [39000 ], [57000 ], [75000 ]
// Tensor(1, 1, 2, 3):
// [[[2 , 3 , 4 ], [5 , 6 , 7 ]]]
// q size: 1
// Backward +
// Backward broadcast
// Backward broadcast
// Tensor(1, 1, 1, 1):
// [[[6 ]]]
// Tensor(2, 3):
// [1 , 1 , 1 ], [1 , 1 , 1 ]