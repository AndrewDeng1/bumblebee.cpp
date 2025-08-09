#include "tensor.h"
#include <iostream>

int main() {
    // ===== POW FUNCTION TEST =====
    auto pow_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
        true
    );

    cout << "POW TEST - Input tensor:" << endl;
    pow_input->print();
    // Expected: Tensor(2, 3): [2, 3, 4], [5, 6, 7]

    auto pow_output = pow_input->pow(2.0f);
    cout << "POW(2) - Squaring each element:" << endl;
    pow_output->print();  
    // Expected: Tensor(2, 3): [4, 9, 16], [25, 36, 49]

    pow_output->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        true
    );

    pow_output->backward();
    cout << "Gradients for POW(2):" << endl;
    pow_input->grad->print();  
    // Expected: Gradients should be 2*x (derivative of x^2 is 2x)
    // Expected: Tensor(2, 3): [4, 6, 8], [10, 12, 14]

    // Test pow with 0.5 (square root)
    auto pow_output_sqrt = pow_input->pow(0.5f);
    cout << "POW(0.5) - Square root of each element:" << endl;
    pow_output_sqrt->print();  
    // Expected: Tensor(2, 3): [1.4142, 1.7321, 2], [2.2361, 2.4495, 2.6458]

    cout << "\n" << string(50, '=') << "\n" << endl;

    // ===== MEAN FUNCTION TEST =====
    auto mean_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "MEAN TEST - Input tensor:" << endl;
    mean_input->print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    auto mean_output_axis1 = mean_input->mean(1, true);
    cout << "MEAN along axis 1 (mean of each row):" << endl;
    mean_output_axis1->print();  
    // Expected: Row 1 mean = (1+2+3)/3 = 2, Row 2 mean = (4+5+6)/3 = 5
    // Expected: Tensor(2, 1): [2], [5]

    mean_output_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 1},
        vector<float>{1.0f, 1.0f},
        true
    );

    mean_output_axis1->backward();
    cout << "Gradients for MEAN along axis 1:" << endl;
    mean_input->grad->print();  
    // Expected: Each element gets 1/N of the gradient (N=3 for axis 1)
    // Expected: Tensor(2, 3): [0.3333, 0.3333, 0.3333], [0.3333, 0.3333, 0.3333]

    auto mean_output_axis0 = mean_input->mean(0, true);
    cout << "MEAN along axis 0 (mean of each column):" << endl;
    mean_output_axis0->print();  
    // Expected: Col 1 mean = (1+4)/2 = 2.5, Col 2 mean = (2+5)/2 = 3.5, Col 3 mean = (3+6)/2 = 4.5
    // Expected: Tensor(1, 3): [2.5, 3.5, 4.5]

    mean_output_axis0->grad = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{1.0f, 1.0f, 1.0f},
        true
    );

    mean_output_axis0->backward();
    cout << "Gradients for MEAN along axis 0:" << endl;
    mean_input->grad->print();  
    // Expected: Each element gets 1/N of the gradient (N=2 for axis 0)
    // Expected: Tensor(2, 3): [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    cout << "\n" << string(50, '=') << "\n" << endl;

    // ===== VARIANCE SQUARED FUNCTION TEST =====
    auto var_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "VARIANCE SQUARED TEST - Input tensor:" << endl;
    var_input->print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    auto var_output_axis1 = var_input->variance_squared(1, true);
    cout << "VARIANCE SQUARED along axis 1 (variance of each row):" << endl;
    var_output_axis1->print();  
    // Expected: Row 1: mean=2, variance = ((1-2)²+(2-2)²+(3-2)²)/3 = (1+0+1)/3 = 0.6667
    // Expected: Row 2: mean=5, variance = ((4-5)²+(5-5)²+(6-5)²)/3 = (1+0+1)/3 = 0.6667
    // Expected: Tensor(2, 1): [0.6667], [0.6667]

    var_output_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 1},
        vector<float>{1.0f, 1.0f},
        true
    );

    var_output_axis1->backward();
    cout << "Gradients for VARIANCE SQUARED along axis 1:" << endl;
    var_input->grad->print();  
    // Expected: Gradients for variance calculation

    auto var_output_axis0 = var_input->variance_squared(0, true);
    cout << "VARIANCE SQUARED along axis 0 (variance of each column):" << endl;
    var_output_axis0->print();  
    // Expected: Col 1: mean=2.5, variance = ((1-2.5)²+(4-2.5)²)/2 = (2.25+2.25)/2 = 2.25
    // Expected: Col 2: mean=3.5, variance = ((2-3.5)²+(5-3.5)²)/2 = (2.25+2.25)/2 = 2.25
    // Expected: Col 3: mean=4.5, variance = ((3-4.5)²+(6-4.5)²)/2 = (2.25+2.25)/2 = 2.25
    // Expected: Tensor(1, 3): [2.25, 2.25, 2.25]

    var_output_axis0->grad = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{1.0f, 1.0f, 1.0f},
        true
    );

    var_output_axis0->backward();
    cout << "Gradients for VARIANCE SQUARED along axis 0:" << endl;
    var_input->grad->print();  
    // Expected: Gradients for variance calculation

    cout << "\n" << string(50, '=') << "\n" << endl;

    // ===== LAYER NORM TEST =====
    auto norm_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "Input tensor:" << endl;
    norm_input->print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    auto norm_output = norm_input->norm(1, true);
    cout << "Layer norm along axis 1 (normalize each row):" << endl;
    norm_output->print();  
    // Expected: Each row normalized to mean=0, std=1
    // Row 1: mean=2, std=1, normalized: [-1.2247, 0, 1.2247]
    // Row 2: mean=5, std=1, normalized: [-1.2247, 0, 1.2247]
    // Expected: Tensor(2, 3): [-1.2247, 0, 1.2247], [-1.2247, 0, 1.2247]

    norm_output->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        true
    );

    norm_output->backward();
    cout << "Gradients for layer norm along axis 1:" << endl;
    norm_input->grad->print();  
    // Expected: Gradients showing how changes in normalized output affect input

    // Test layer norm along axis 0 (normalize each column)
    auto norm_output_axis0 = norm_input->norm(0, true);
    cout << "Layer norm along axis 0 (normalize each column):" << endl;
    norm_output_axis0->print();  
    // Expected: Each column normalized to mean=0, std=1
    // Col 1: mean=2.5, std=2.1213, normalized: [-0.7071, 0.7071]
    // Col 2: mean=3.5, std=2.1213, normalized: [-0.7071, 0.7071]
    // Col 3: mean=4.5, std=2.1213, normalized: [-0.7071, 0.7071]
    // Expected: Tensor(2, 3): [-0.7071, -0.7071, -0.7071], [0.7071, 0.7071, 0.7071]

    norm_output_axis0->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        true
    );

    norm_output_axis0->backward();
    cout << "Gradients for layer norm along axis 0:" << endl;
    norm_input->grad->print();  
    // Expected: Gradients showing how changes in normalized output affect input

    // Test with different values to see normalization effect
    auto norm_input2 = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{10.0f, 20.0f, 30.0f, 40.0f},
        true
    );

    cout << "\nSecond test - Input tensor:" << endl;
    norm_input2->print();
    // Expected: Tensor(2, 2): [10, 20], [30, 40]

    auto norm_output2 = norm_input2->norm(1, true);
    cout << "Layer norm along axis 1:" << endl;
    norm_output2->print();  
    // Expected: Each row normalized independently
    // Row 1: mean=15, std=7.0711, normalized: [-0.7071, 0.7071]
    // Row 2: mean=35, std=7.0711, normalized: [-0.7071, 0.7071]
    // Expected: Tensor(2, 2): [-0.7071, 0.7071], [-0.7071, 0.7071]

    return 0;
}



// POW TEST - Input tensor:
// Tensor(2, 3):
// [2 , 3 , 4 ], [5 , 6 , 7 ]
// POW(2) - Squaring each element:
// Tensor(2, 3):
// [4 , 9 , 16 ], [25 , 36 , 49 ]
// Gradients for POW(2):
// Tensor(2, 3):
// [4 , 6 , 8 ], [10 , 12 , 14 ]
// POW(0.5) - Square root of each element:
// Tensor(2, 3):
// [1.41421 , 1.73205 , 2 ], [2.23607 , 2.44949 , 2.64575 ]

// ==================================================

// MEAN TEST - Input tensor:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// MEAN along axis 1 (mean of each row):
// Tensor(2, 1):
// [2 ], [5 ]
// Gradients for MEAN along axis 1:
// Tensor(2, 3):
// [0.333333 , 0.333333 , 0.333333 ], [0.333333 , 0.333333 , 0.333333 ]
// MEAN along axis 0 (mean of each column):
// Tensor(1, 3):
// [2.5 , 3.5 , 4.5 ]
// Gradients for MEAN along axis 0:
// Tensor(2, 3):
// [0.833333 , 0.833333 , 0.833333 ], [0.833333 , 0.833333 , 0.833333 ]

// ==================================================

// VARIANCE SQUARED TEST - Input tensor:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// VARIANCE SQUARED along axis 1 (variance of each row):
// Tensor(2, 1):
// [0.666667 ], [0.666667 ]
// Gradients for VARIANCE SQUARED along axis 1:
// Tensor(2, 3):
// [-0.666667, 0 , 0.666667 ], [-0.666667, 0 , 0.666667 ]
// VARIANCE SQUARED along axis 0 (variance of each column):
// Tensor(1, 3):
// [2.25 , 2.25 , 2.25 ]
// Gradients for VARIANCE SQUARED along axis 0:
// Tensor(2, 3):
// [-2.16667, -1.5, -0.833333], [0.833333 , 1.5 , 2.16667 ]

// ==================================================

// Input tensor:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// Layer norm along axis 1 (normalize each row):
// Tensor(2, 3):
// [-1.22474, 0 , 1.22474 ], [-1.22474, 0 , 1.22474 ]
// Gradients for layer norm along axis 1:
// Tensor(2, 3):
// [2.44947 , 2.44947 , 2.44947 ], [2.44947 , 2.44947 , 2.44947 ]
// Layer norm along axis 0 (normalize each column):
// Tensor(2, 3):
// [-0.999998, -0.999998, -0.999998], [0.999998 , 0.999998 , 0.999998 ]
// Gradients for layer norm along axis 0:
// Tensor(2, 3):
// [3.7828 , 3.7828 , 3.7828 ], [3.7828 , 3.7828 , 3.7828 ]

// Second test - Input tensor:
// Tensor(2, 2):
// [10 , 20 ], [30 , 40 ]
// Layer norm along axis 1:
// Tensor(2, 2):
// [-1, 1 ], [-1, 1 ]