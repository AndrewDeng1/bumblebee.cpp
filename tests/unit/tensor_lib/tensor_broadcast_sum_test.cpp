#include "tensor.h"
#include <iostream>

int main() {

    vector<int> shape = {2, 1, 3};
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t1 = make_shared<Tensor>(shape, data, true);

    t1->print();
    cout<<"we got here"<<endl;
    auto t2 = t1->broadcast({2, 4, 3});
    t2->print();

    t2->grad = make_shared<Tensor>(vector<int>{2, 4, 3}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false);
    t2->backward_fn();

    t1->grad->print();

    auto t3 = t1->sum(1, true);
    t3->print();
    // t1->grad->print();
    t3->grad = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false);
    t3->grad->print();
    t3->backward_fn();
    t1->grad->print();

    vector<int> shape1 = {2, 2, 3};
    vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    auto t4 = make_shared<Tensor>(shape1, data1, true);
    t4->print();

    auto t5 = t4->sum(1, true);
    t5->print();
    t5->grad = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false);
    t5->backward_fn();
    t4->grad->print();

    auto t6 = t4->sum(0, true);
    t6->print();
    t6->grad = make_shared<Tensor>(vector<int>{1, 2, 3}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false);
    t6->backward_fn();
    t4->grad->print();

    auto t7 = t4->sum(2, true);
    t7->print();
    t7->grad = make_shared<Tensor>(vector<int>{2, 2, 1}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, false);
    t7->backward_fn();
    t4->grad->print();

    auto t8 = t7->broadcast({2, 2, 3});
    t8->print();
    t8->grad = make_shared<Tensor>(vector<int>{2, 2, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, false);
    t8->backward_fn();
    t7->grad->print();


    // // // Create a 2x3 tensor
    // vector<int> shape = {2, 4, 3};
    // vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    // auto t1 = make_shared<Tensor>(shape, data, true);
    // // cout << t1->size() << endl;
    // // cout << t1->at({0, 0, 0}) << endl;
    // // cout << t1->at({1, 0, 0}) << endl;
    // // cout << t1->at({0, 1, 0}) << endl;
    // // cout << t1->at({1, 1, 0}) << endl;
    // // cout << t1->at({0, 2, 0}) << endl;
    // // cout << t1->at({1, 2, 0}) << endl;
    // // cout << t1->at({0, 3, 0}) << endl;
    // // cout << t1->at({1, 3, 0}) << endl;
    // // cout << t1->at({0, 0, 0, 2}) << endl;
    // auto t2 = t1->reshape({2, 3, 4});
    // auto t3 = t2->reshape({2, 12});
    // // cout << t3->at({0, 0}) << endl;   
    // // cout << t3->at({0, 1}) << endl;
    // // cout << t3->at({0, 2}) << endl;
    // // cout << t3->at({0, 3}) << endl;
    // // cout << t3->at({0, 4}) << endl;
    // // cout << t3->at({0, 5}) << endl;
    // // cout << t3->at({0, 6}) << endl;
    // // cout << t3->at({0, 7}) << endl;
    // // cout << t3->at({0, 8}) << endl;
    // // cout << t3->at({0, 9}) << endl;
    // // cout << t3->at({0, 10}) << endl;
    // // cout << t3->at({0, 11}) << endl;
    // // cout << t3->at({1, 0}) << endl;   
    // // cout << t3->at({1, 1}) << endl;
    // // cout << t3->at({1, 2}) << endl;
    // // cout << t3->at({1, 3}) << endl;
    // // cout << t3->at({1, 4}) << endl;
    // // cout << t3->at({1, 5}) << endl;
    // // cout << t3->at({1, 6}) << endl;
    // // cout << t3->at({1, 7}) << endl;
    // // cout << t3->at({1, 8}) << endl;
    // // cout << t3->at({1, 9}) << endl;
    // // cout << t3->at({1, 10}) << endl;
    // // cout << t3->at({1, 11}) << endl;
    // auto t4 = t3->sum(0, true);
    // // cout << t4->shape[0] << endl;
    // // cout << t4->shape[1] << endl;
    // // cout << t4->at({0, 0}) << endl;
    // // cout << t4->at({0, 1}) << endl;
    // // cout << t4->at({0, 2}) << endl;
    // // cout << t4->at({0, 3}) << endl;
    // // cout << t4->at({0, 4}) << endl;
    // // cout << t4->at({0, 5}) << endl;
    // // cout << t4->at({0, 6}) << endl;
    // // cout << t4->at({0, 7}) << endl;
    // // cout << t4->at({0, 8}) << endl;
    // // cout << t4->at({0, 9}) << endl;
    // // cout << t4->at({0, 10}) << endl;
    // // cout << t4->at({0, 11}) << endl;
    // auto t5 = t3->sum(1, true);
    // // cout << t5->shape[0] << endl;
    // // cout << t5->shape[1] << endl;
    // // cout << t5->at({0, 0}) << endl;
    // // cout << t5->at({1, 0}) << endl;

    // int sm1=0;
    // int sm2=0;
    // for(int i=0; i<data.size()/2; i++){
    //     sm1+=data[i];
    //     sm2+=data[i+data.size()/2];
    // }
    // // cout << sm1 << endl;
    // // cout << sm2 << endl;

    // auto t6 = t1->sum(1, true);
    // // printf("t6 shape: %d, %d, %d\n", t6->shape[0], t6->shape[1], t6->shape[2]);
    
    // // for(int i=0; i<t6->shape[0]; i++){
    // //     for(int j=0; j<t6->shape[2]; j++){
    // //         cout<<t6->at({i, 0, j})<<endl;
    // //     }
    // // }

    // auto t7 = t1->reduce_to_shape({2, 1, 1});
    // // printf("t7 shape: %d, %d, %d\n", t7->shape[0], t7->shape[1], t7->shape[2]);
    // // cout<<t7->at({0, 0, 0})<<endl;
    // // cout<<t7->at({1, 0, 0})<<endl;
    
    // // printf("t1 shape: %d, %d, %d\n", t1->shape[0], t1->shape[1], t1->shape[2]);
    // // printf("t7 shape: %d, %d, %d\n", t7->shape[0], t7->shape[1], t7->shape[2]);
    // auto t8 = t1 + t7;

    // auto a1 = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 100.0f, 1000.0f, 750.0f, 500.0f, 5000.0f}, true);
    // a1->print();
    // t1->print();
    // auto a2 = a1+t1;
    // // printf("a2 shape: %d, %d, %d\n", a2->shape[0], a2->shape[1], a2->shape[2]);
    // // cout << t1->at({0, 0, 0}) << endl;
    // // cout << t1->at({1, 0, 0}) << endl;
    // // cout << t1->at({0, 1, 0}) << endl;
    // // cout << t1->at({1, 1, 0}) << endl;
    // // cout << t1->at({0, 2, 0}) << endl;
    // // cout << t1->at({1, 2, 0}) << endl;
    // // cout << t1->at({0, 3, 0}) << endl;
    // // cout << t1->at({1, 3, 0}) << endl;
    // a2->print();
    // auto a3 = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    // auto a4 = make_shared<Tensor>(vector<int>{3, 1}, vector<float>{100.0f, 200.0f, 300.0f}, true);
    // auto a5 = a3+a4;
    // a5->print();
    // a5->grad = make_shared<Tensor>(vector<int>{2, 3, 3}, vector<float>{1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f, 6000.0f, 7000.0f, 8000.0f, 9000.0f, 10000.0f, 11000.0f, 12000.0f, 13000.0f, 14000.0f, 15000.0f, 16000.0f, 17000.0f, 18000.0f}, true);
    // a5->grad->print();
    // a5->backward_fn();
    // a3->grad->print();
    // a4->grad->print();

    // // grad
    // // 1000 2000 3000   10000 11000 12000
    // // 4000 5000 6000   13000 14000 15000
    // // 7000 8000 9000   16000 17000 18000

    // // a3->grad
    // // 5000+4000+7000 2000+5000+8000 3000+6000+9000    10000+13000+16000 11000+14000+17000 12000+15000+18000
    // // = 16000 15000 18000    33000 42000 45000

    // // a4->grad
    // // (1000+10000) + (2000+11000) + (3000+12000)
    // // (4000+13000) + (5000+14000) + (6000+15000)
    // // (7000+16000) + (8000+17000) + (9000+18000)
    // // = 11000+13000+15000
    // // = 17000+19000+21000
    // // = 23000+25000+27000
    // // = 39000
    // // = 57000
    // // = 75000

    // auto a6 = make_shared<Tensor>(vector<int>{1, 1, 1, 1}, vector<float>{1.0f}, true);
    // auto a7 = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);

    // // a6+=a7;
    // // auto a8=a6;
    // auto a8 = a7-a6;
    // a8->print();
    // a8->grad = make_shared<Tensor>(vector<int>{1, 1, 2, 3}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, true);
    // a8->backward_fn();
    // a6->grad->print();
    // a7->grad->print();

    // a8->print();
    // auto a9 = a8->transpose(-2, -1);
    // a9->print();
    // a9->grad = make_shared<Tensor>(vector<int>{1, 1, 3, 2}, vector<float>{10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f}, true);
    // a9->backward_fn();
    // a8->grad->print();

    
}

// CORRECT OUTPUT

// Tensor(2, 1, 3):
// [[1 , 2 , 3 ]], [[4 , 5 , 6 ]]
// we got here
// Tensor(2, 4, 3):
// [[1 , 2 , 3 ], [1 , 2 , 3 ], [1 , 2 , 3 ], [1 , 2 , 3 ]], [[4 , 5 , 6 ], [4 , 5 , 6 ], [4 , 5 , 6 ], [4 , 5 , 6 ]]
// Tensor(2, 1, 3):
// [[4 , 4 , 4 ]], [[4 , 4 , 4 ]]
// Tensor(2, 1, 3):
// [[1 , 2 , 3 ]], [[4 , 5 , 6 ]]
// Tensor(2, 1, 3):
// [[1 , 1 , 1 ]], [[1 , 1 , 1 ]]
// Tensor(2, 1, 3):
// [[5 , 5 , 5 ]], [[5 , 5 , 5 ]]
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [4 , 5 , 6 ]], [[7 , 8 , 9 ], [10 , 11 , 12 ]]
// Tensor(2, 1, 3):
// [[5 , 7 , 9 ]], [[11 , 13 , 15 ]]
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [1 , 2 , 3 ]], [[4 , 5 , 6 ], [4 , 5 , 6 ]]
// Tensor(1, 2, 3):
// [[8 , 10 , 12 ], [14 , 16 , 18 ]]
// Tensor(2, 2, 3):
// [[2 , 3 , 4 ], [2 , 3 , 4 ]], [[5 , 6 , 7 ], [5 , 6 , 7 ]]
// Tensor(2, 2, 1):
// [[6 ], [9 ]], [[12 ], [15 ]]
// Tensor(2, 2, 3):
// [[3 , 4 , 5 ], [4 , 5 , 6 ]], [[8 , 9 , 10 ], [9 , 10 , 11 ]]
// Tensor(2, 2, 3):
// [[6 , 6 , 6 ], [9 , 9 , 9 ]], [[12 , 12 , 12 ], [15 , 15 , 15 ]]
// Tensor(2, 2, 1):
// [[7 ], [11 ]], [[15 ], [19 ]]