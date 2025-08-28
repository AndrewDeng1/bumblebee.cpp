#include "tensor.h"
#include <iostream>

int main() {
    // ===== SOFTMAX FUNCTION TEST =====
    auto softmax_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "SOFTMAX TEST - Input tensor:" << endl;
    softmax_input->print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    auto softmax_output_axis1 = softmax_input->softmax(1, true);
    cout << "SOFTMAX along axis 1 (softmax of each row):" << endl;
    softmax_output_axis1->print();  
    // Expected: Row 1: exp(1)=2.718, exp(2)=7.389, exp(3)=20.086, sum=30.193
    // Expected: Row 1: [0.0900, 0.2447, 0.6652]
    // Expected: Row 2: exp(4)=54.598, exp(5)=148.413, exp(6)=403.429, sum=606.440
    // Expected: Row 2: [0.0900, 0.2447, 0.6652]
    // Expected: Tensor(2, 3): [0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]

    softmax_output_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
        true
    );

    softmax_output_axis1->backward();
    cout << "Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    auto softmax_output_axis0 = softmax_input->softmax(0, true);
    cout << "SOFTMAX along axis 0 (softmax of each column):" << endl;
    softmax_output_axis0->print();  
    // Expected: Col 1: exp(1)=2.718, exp(4)=54.598, sum=57.316
    // Expected: Col 1: [0.0474, 0.9526]
    // Expected: Col 2: exp(2)=7.389, exp(5)=148.413, sum=155.802
    // Expected: Col 2: [0.0474, 0.9526]
    // Expected: Col 3: exp(3)=20.086, exp(6)=403.429, sum=423.515
    // Expected: Col 3: [0.0474, 0.9526]
    // Expected: Tensor(2, 3): [0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]

    softmax_output_axis0->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    softmax_output_axis0->backward();
    cout << "Gradients for SOFTMAX along axis 0:" << endl;
    softmax_input->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    // Test with negative values to see softmax behavior
    auto softmax_input2 = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{-2.0f, 2.0f, -1.0f, 1.0f},
        true
    );

    cout << "\nSecond SOFTMAX test - Input tensor with negative values:" << endl;
    softmax_input2->print();
    // Expected: Tensor(2, 2): [-2, 2], [-1, 1]

    auto softmax_output2 = softmax_input2->softmax(1, true);
    cout << "SOFTMAX along axis 1 (with negative values):" << endl;
    softmax_output2->print();  
    // Expected: Row 1: exp(-2)=0.1353, exp(2)=7.3891, sum=7.5244
    // Expected: Row 1: [0.0180, 0.9820]
    // Expected: Row 2: exp(-1)=0.3679, exp(1)=2.7183, sum=3.0862
    // Expected: Row 2: [0.1192, 0.8808]
    // Expected: Tensor(2, 2): [0.0180, 0.9820], [0.1192, 0.8808]

    softmax_output2->grad = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{1.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    softmax_output2->backward();
    cout << "Gradients for SOFTMAX with negative values:" << endl;
    softmax_input2->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    // Test with large values to check numerical stability
    auto softmax_input3 = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{100.0f, 101.0f, 102.0f},
        true
    );

    cout << "\nThird SOFTMAX test - Input tensor with large values:" << endl;
    softmax_input3->print();
    // Expected: Tensor(1, 3): [100, 101, 102]

    auto softmax_output3 = softmax_input3->softmax(1, true);
    cout << "SOFTMAX with large values (should be numerically stable):" << endl;
    softmax_output3->print();  
    // Expected: Should handle large values without overflow
    // Expected: exp(100), exp(101), exp(102) are very large but ratios should be correct
    // Expected: Tensor(1, 3): [0.0900, 0.2447, 0.6652] (approximate ratios)

    softmax_output3->grad = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{1.0f, 0.0f, 0.0f},
        true
    );

    softmax_output3->backward();
    cout << "Gradients for SOFTMAX with large values:" << endl;
    softmax_input3->grad->print();  
    // Expected: Gradients should be computed correctly even with large input values

    // ===== HIGHER DIMENSION SOFTMAX TEST =====
    cout << "\n" << string(50, '=') << endl;
    cout << "HIGHER DIMENSION SOFTMAX TEST" << endl;
    cout << string(50, '=') << endl;

    // Create 3D input tensor (batch_size=2, sequence_length=3, num_classes=4)
    auto softmax_input_3d = make_shared<Tensor>(
        vector<int>{2, 3, 4},
        vector<float>{
            // Batch 0, Sequence 0: [0.1, 0.2, 0.3, 0.4]
            0.1f, 0.2f, 0.3f, 0.4f,
            // Batch 0, Sequence 1: [0.5, 0.6, 0.7, 0.8]
            0.5f, 0.6f, 0.7f, 0.8f,
            // Batch 0, Sequence 2: [0.9, 1.0, 1.1, 1.2]
            0.9f, 1.0f, 1.1f, 1.2f,
            // Batch 1, Sequence 0: [1.3, 1.4, 1.5, 1.6]
            1.3f, 1.4f, 1.5f, 1.6f,
            // Batch 1, Sequence 1: [1.7, 1.8, 1.9, 2.0]
            1.7f, 1.8f, 1.9f, 2.0f,
            // Batch 1, Sequence 2: [2.1, 2.2, 2.3, 2.4]
            2.1f, 2.2f, 2.3f, 2.4f
        },
        true
    );

    cout << "3D Input tensor shape: [" << softmax_input_3d->shape[0] << ", " 
         << softmax_input_3d->shape[1] << ", " << softmax_input_3d->shape[2] << "]" << endl;
    cout << "3D Input tensor:" << endl;
    softmax_input_3d->print();

    // Test softmax along the last axis (axis 2) - class probabilities
    auto softmax_output_3d_axis2 = softmax_input_3d->softmax(2, true);
    cout << "3D SOFTMAX along axis 2 (class probabilities):" << endl;
    softmax_output_3d_axis2->print();
    // Expected: Should compute softmax for each sequence position across classes

    softmax_output_3d_axis2->grad = make_shared<Tensor>(
        vector<int>{2, 3, 4},
        vector<float>{
            1.0f, 0.0f, 0.0f, 0.0f,  // Batch 0, Sequence 0
            0.0f, 1.0f, 0.0f, 0.0f,  // Batch 0, Sequence 1
            0.0f, 0.0f, 1.0f, 0.0f,  // Batch 0, Sequence 2
            0.0f, 0.0f, 0.0f, 1.0f,  // Batch 1, Sequence 0
            1.0f, 0.0f, 0.0f, 0.0f,  // Batch 1, Sequence 1
            0.0f, 1.0f, 0.0f, 0.0f   // Batch 1, Sequence 2
        },
        true
    );

    softmax_output_3d_axis2->backward();
    cout << "3D Gradients for SOFTMAX along axis 2:" << endl;
    softmax_input_3d->grad->print();

    // Test softmax along the sequence axis (axis 1) - attention weights
    auto softmax_output_3d_axis1 = softmax_input_3d->softmax(1, true);
    cout << "3D SOFTMAX along axis 1 (attention weights):" << endl;
    softmax_output_3d_axis1->print();
    // Expected: Should compute softmax across sequence positions for each class

    softmax_output_3d_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3, 4},
        vector<float>{
            1.0f, 1.0f, 1.0f, 1.0f,  // Batch 0, Sequence 0
            0.0f, 0.0f, 0.0f, 0.0f,  // Batch 0, Sequence 1
            0.0f, 0.0f, 0.0f, 0.0f,  // Batch 0, Sequence 2
            0.0f, 0.0f, 0.0f, 0.0f,  // Batch 1, Sequence 0
            1.0f, 1.0f, 1.0f, 1.0f,  // Batch 1, Sequence 1
            0.0f, 0.0f, 0.0f, 0.0f   // Batch 1, Sequence 2
        },
        true
    );

    softmax_output_3d_axis1->backward();
    cout << "3D Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input_3d->grad->print();

    // ===== 4D TENSOR TEST (BATCH, CHANNELS, HEIGHT, WIDTH) =====
    cout << "\n" << string(50, '=') << endl;
    cout << "4D TENSOR SOFTMAX TEST" << endl;
    cout << string(50, '=') << endl;

    // Create 4D input tensor (batch_size=2, channels=3, height=2, width=2)
    auto softmax_input_4d = make_shared<Tensor>(
        vector<int>{2, 3, 2, 2},
        vector<float>{
            // Batch 0, Channel 0
            0.1f, 0.2f, 0.3f, 0.4f,
            // Batch 0, Channel 1
            0.5f, 0.6f, 0.7f, 0.8f,
            // Batch 0, Channel 2
            0.9f, 1.0f, 1.1f, 1.2f,
            // Batch 1, Channel 0
            1.3f, 1.4f, 1.5f, 1.6f,
            // Batch 1, Channel 1
            1.7f, 1.8f, 1.9f, 2.0f,
            // Batch 1, Channel 2
            2.1f, 2.2f, 2.3f, 2.4f
        },
        true
    );

    cout << "4D Input tensor shape: [" << softmax_input_4d->shape[0] << ", " 
         << softmax_input_4d->shape[1] << ", " << softmax_input_4d->shape[2] << ", " 
         << softmax_input_4d->shape[3] << "]" << endl;
    cout << "4D Input tensor:" << endl;
    softmax_input_4d->print();

    // Test softmax along the channel axis (axis 1) - multi-class segmentation
    auto softmax_output_4d_axis1 = softmax_input_4d->softmax(1, true);
    cout << "4D SOFTMAX along axis 1 (channel probabilities):" << endl;
    softmax_output_4d_axis1->print();
    // Expected: Should compute softmax across channels for each spatial position

    softmax_output_4d_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3, 2, 2},
        vector<float>{
            // Batch 0, Channel 0
            1.0f, 0.0f, 0.0f, 1.0f,
            // Batch 0, Channel 1
            0.0f, 1.0f, 0.0f, 0.0f,
            // Batch 0, Channel 2
            0.0f, 0.0f, 1.0f, 0.0f,
            // Batch 1, Channel 0
            0.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 1
            1.0f, 0.0f, 0.0f, 1.0f,
            // Batch 1, Channel 2
            0.0f, 0.0f, 1.0f, 0.0f
        },
        true
    );

    softmax_output_4d_axis1->backward();
    cout << "4D Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input_4d->grad->print();

    // Test softmax along the spatial axes (axis 2 and 3) - spatial attention
    auto softmax_output_4d_axis2 = softmax_input_4d->softmax(2, true);
    cout << "4D SOFTMAX along axis 2 (height attention):" << endl;
    softmax_output_4d_axis2->print();
    // Expected: Should compute softmax across height dimension

    softmax_output_4d_axis2->grad = make_shared<Tensor>(
        vector<int>{2, 3, 2, 2},
        vector<float>{
            // Batch 0, Channel 0
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 0, Channel 1
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 0, Channel 2
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 0
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 1
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 2
            1.0f, 1.0f, 0.0f, 0.0f
        },
        true
    );

    softmax_output_4d_axis2->backward();
    cout << "4D Gradients for SOFTMAX along axis 2:" << endl;
    softmax_input_4d->grad->print();

    return 0;
}



// SOFTMAX TEST - Input tensor:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// SOFTMAX along axis 1 (softmax of each row):
// Tensor(2, 3):
// [0.0900306 , 0.244728 , 0.665241 ], [0.0900306 , 0.244728 , 0.665241 ]
// Gradients for SOFTMAX along axis 1:
// Tensor(2, 3):
// [0.0819251 , -0.022033, -0.059892], [-0.022033, 0.184836 , -0.162803]
// SOFTMAX along axis 0 (softmax of each column):
// Tensor(2, 3):
// [0.0474259 , 0.0474259 , 0.0474259 ], [0.952574 , 0.952574 , 0.952574 ]
// Gradients for SOFTMAX along axis 0:
// Tensor(2, 3):
// [0.127102 , -0.022033, -0.105069], [-0.0672097, 0.184836 , -0.117627]

// Second SOFTMAX test - Input tensor with negative values:
// Tensor(2, 2):
// [-2, 2 ], [-1, 1 ]
// SOFTMAX along axis 1 (with negative values):
// Tensor(2, 2):
// [0.0179862 , 0.982014 ], [0.119203 , 0.880797 ]
// Gradients for SOFTMAX with negative values:
// Tensor(2, 2):
// [0.0176627 , -0.0176627], [-0.104994, 0.104994 ]

// Third SOFTMAX test - Input tensor with large values:
// Tensor(1, 3):
// [100 , 101 , 102 ]
// SOFTMAX with large values (should be numerically stable):
// Tensor(1, 3):
// [0.0900306 , 0.244728 , 0.665241 ]
// Gradients for SOFTMAX with large values:
// Tensor(1, 3):
// [0.0819251 , -0.022033, -0.059892]


#include "tensor.h"
#include <iostream>

int main() {
    // ===== SOFTMAX FUNCTION TEST =====
    auto softmax_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "SOFTMAX TEST - Input tensor:" << endl;
    softmax_input->print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    auto softmax_output_axis1 = softmax_input->softmax(1);
    cout << "SOFTMAX along axis 1 (softmax of each row):" << endl;
    softmax_output_axis1->print();  
    // Expected: Row 1: exp(1)=2.718, exp(2)=7.389, exp(3)=20.086, sum=30.193
    // Expected: Row 1: [0.0900, 0.2447, 0.6652]
    // Expected: Row 2: exp(4)=54.598, exp(5)=148.413, exp(6)=403.429, sum=606.440
    // Expected: Row 2: [0.0900, 0.2447, 0.6652]
    // Expected: Tensor(2, 3): [0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]

    softmax_output_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
        true
    );

    softmax_output_axis1->backward();
    cout << "Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    auto softmax_output_axis0 = softmax_input->softmax(0);
    cout << "SOFTMAX along axis 0 (softmax of each column):" << endl;
    softmax_output_axis0->print();  
    // Expected: Col 1: exp(1)=2.718, exp(4)=54.598, sum=57.316
    // Expected: Col 1: [0.0474, 0.9526]
    // Expected: Col 2: exp(2)=7.389, exp(5)=148.413, sum=155.802
    // Expected: Col 2: [0.0474, 0.9526]
    // Expected: Col 3: exp(3)=20.086, exp(6)=403.429, sum=423.515
    // Expected: Col 3: [0.0474, 0.9526]
    // Expected: Tensor(2, 3): [0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]

    softmax_output_axis0->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    softmax_output_axis0->backward();
    cout << "Gradients for SOFTMAX along axis 0:" << endl;
    softmax_input->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    // Test with negative values to see softmax behavior
    auto softmax_input2 = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{-2.0f, 2.0f, -1.0f, 1.0f},
        true
    );

    cout << "\nSecond SOFTMAX test - Input tensor with negative values:" << endl;
    softmax_input2->print();
    // Expected: Tensor(2, 2): [-2, 2], [-1, 1]

    auto softmax_output2 = softmax_input2->softmax(1);
    cout << "SOFTMAX along axis 1 (with negative values):" << endl;
    softmax_output2->print();  
    // Expected: Row 1: exp(-2)=0.1353, exp(2)=7.3891, sum=7.5244
    // Expected: Row 1: [0.0180, 0.9820]
    // Expected: Row 2: exp(-1)=0.3679, exp(1)=2.7183, sum=3.0862
    // Expected: Row 2: [0.1192, 0.8808]
    // Expected: Tensor(2, 2): [0.0180, 0.9820], [0.1192, 0.8808]

    softmax_output2->grad = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{1.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    softmax_output2->backward();
    cout << "Gradients for SOFTMAX with negative values:" << endl;
    softmax_input2->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    // Test with large values to check numerical stability
    auto softmax_input3 = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{100.0f, 101.0f, 102.0f},
        true
    );

    cout << "\nThird SOFTMAX test - Input tensor with large values:" << endl;
    softmax_input3->print();
    // Expected: Tensor(1, 3): [100, 101, 102]

    auto softmax_output3 = softmax_input3->softmax(1);
    cout << "SOFTMAX with large values (should be numerically stable):" << endl;
    softmax_output3->print();  
    // Expected: Should handle large values without overflow
    // Expected: exp(100), exp(101), exp(102) are very large but ratios should be correct
    // Expected: Tensor(1, 3): [0.0900, 0.2447, 0.6652] (approximate ratios)

    softmax_output3->grad = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{1.0f, 0.0f, 0.0f},
        true
    );

    softmax_output3->backward();
    cout << "Gradients for SOFTMAX with large values:" << endl;
    softmax_input3->grad->print();  
    // Expected: Gradients should be computed correctly even with large input values

    // ===== HIGHER DIMENSION SOFTMAX TEST =====
    cout << "\n" << string(50, '=') << endl;
    cout << "HIGHER DIMENSION SOFTMAX TEST" << endl;
    cout << string(50, '=') << endl;

    // Create 3D input tensor (batch_size=2, sequence_length=3, num_classes=4)
    auto softmax_input_3d = make_shared<Tensor>(
        vector<int>{2, 3, 4},
        vector<float>{
            // Batch 0, Sequence 0: [0.1, 0.2, 0.3, 0.4]
            0.1f, 0.2f, 0.3f, 0.4f,
            // Batch 0, Sequence 1: [0.5, 0.6, 0.7, 0.8]
            0.5f, 0.6f, 0.7f, 0.8f,
            // Batch 0, Sequence 2: [0.9, 1.0, 1.1, 1.2]
            0.9f, 1.0f, 1.1f, 1.2f,
            // Batch 1, Sequence 0: [1.3, 1.4, 1.5, 1.6]
            1.3f, 1.4f, 1.5f, 1.6f,
            // Batch 1, Sequence 1: [1.7, 1.8, 1.9, 2.0]
            1.7f, 1.8f, 1.9f, 2.0f,
            // Batch 1, Sequence 2: [2.1, 2.2, 2.3, 2.4]
            2.1f, 2.2f, 2.3f, 2.4f
        },
        true
    );

    cout << "3D Input tensor shape: [" << softmax_input_3d->shape[0] << ", " 
         << softmax_input_3d->shape[1] << ", " << softmax_input_3d->shape[2] << "]" << endl;
    cout << "3D Input tensor:" << endl;
    softmax_input_3d->print();

    // Test softmax along the last axis (axis 2) - class probabilities
    auto softmax_output_3d_axis2 = softmax_input_3d->softmax(2);
    cout << "3D SOFTMAX along axis 2 (class probabilities):" << endl;
    softmax_output_3d_axis2->print();
    // Expected: Should compute softmax for each sequence position across classes

    softmax_output_3d_axis2->grad = make_shared<Tensor>(
        vector<int>{2, 3, 4},
        vector<float>{
            1.0f, 0.0f, 0.0f, 0.0f,  // Batch 0, Sequence 0
            0.0f, 1.0f, 0.0f, 0.0f,  // Batch 0, Sequence 1
            0.0f, 0.0f, 1.0f, 0.0f,  // Batch 0, Sequence 2
            0.0f, 0.0f, 0.0f, 1.0f,  // Batch 1, Sequence 0
            1.0f, 0.0f, 0.0f, 0.0f,  // Batch 1, Sequence 1
            0.0f, 1.0f, 0.0f, 0.0f   // Batch 1, Sequence 2
        },
        true
    );

    softmax_output_3d_axis2->backward();
    cout << "3D Gradients for SOFTMAX along axis 2:" << endl;
    softmax_input_3d->grad->print();

    // Test softmax along the sequence axis (axis 1) - attention weights
    auto softmax_output_3d_axis1 = softmax_input_3d->softmax(1);
    cout << "3D SOFTMAX along axis 1 (attention weights):" << endl;
    softmax_output_3d_axis1->print();
    // Expected: Should compute softmax across sequence positions for each class

    softmax_output_3d_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3, 4},
        vector<float>{
            1.0f, 1.0f, 1.0f, 1.0f,  // Batch 0, Sequence 0
            0.0f, 0.0f, 0.0f, 0.0f,  // Batch 0, Sequence 1
            0.0f, 0.0f, 0.0f, 0.0f,  // Batch 0, Sequence 2
            0.0f, 0.0f, 0.0f, 0.0f,  // Batch 1, Sequence 0
            1.0f, 1.0f, 1.0f, 1.0f,  // Batch 1, Sequence 1
            0.0f, 0.0f, 0.0f, 0.0f   // Batch 1, Sequence 2
        },
        true
    );

    softmax_output_3d_axis1->backward();
    cout << "3D Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input_3d->grad->print();

    // ===== 4D TENSOR TEST (BATCH, CHANNELS, HEIGHT, WIDTH) =====
    cout << "\n" << string(50, '=') << endl;
    cout << "4D TENSOR SOFTMAX TEST" << endl;
    cout << string(50, '=') << endl;

    // Create 4D input tensor (batch_size=2, channels=3, height=2, width=2)
    auto softmax_input_4d = make_shared<Tensor>(
        vector<int>{2, 3, 2, 2},
        vector<float>{
            // Batch 0, Channel 0
            0.1f, 0.2f, 0.3f, 0.4f,
            // Batch 0, Channel 1
            0.5f, 0.6f, 0.7f, 0.8f,
            // Batch 0, Channel 2
            0.9f, 1.0f, 1.1f, 1.2f,
            // Batch 1, Channel 0
            1.3f, 1.4f, 1.5f, 1.6f,
            // Batch 1, Channel 1
            1.7f, 1.8f, 1.9f, 2.0f,
            // Batch 1, Channel 2
            2.1f, 2.2f, 2.3f, 2.4f
        },
        true
    );

    cout << "4D Input tensor shape: [" << softmax_input_4d->shape[0] << ", " 
         << softmax_input_4d->shape[1] << ", " << softmax_input_4d->shape[2] << ", " 
         << softmax_input_4d->shape[3] << "]" << endl;
    cout << "4D Input tensor:" << endl;
    softmax_input_4d->print();

    // Test softmax along the channel axis (axis 1) - multi-class segmentation
    auto softmax_output_4d_axis1 = softmax_input_4d->softmax(1);
    cout << "4D SOFTMAX along axis 1 (channel probabilities):" << endl;
    softmax_output_4d_axis1->print();
    // Expected: Should compute softmax across channels for each spatial position

    softmax_output_4d_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3, 2, 2},
        vector<float>{
            // Batch 0, Channel 0
            1.0f, 0.0f, 0.0f, 1.0f,
            // Batch 0, Channel 1
            0.0f, 1.0f, 0.0f, 0.0f,
            // Batch 0, Channel 2
            0.0f, 0.0f, 1.0f, 0.0f,
            // Batch 1, Channel 0
            0.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 1
            1.0f, 0.0f, 0.0f, 1.0f,
            // Batch 1, Channel 2
            0.0f, 0.0f, 1.0f, 0.0f
        },
        true
    );

    softmax_output_4d_axis1->backward();
    cout << "4D Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input_4d->grad->print();

    // Test softmax along the spatial axes (axis 2 and 3) - spatial attention
    auto softmax_output_4d_axis2 = softmax_input_4d->softmax(2);
    cout << "4D SOFTMAX along axis 2 (height attention):" << endl;
    softmax_output_4d_axis2->print();
    // Expected: Should compute softmax across height dimension

    softmax_output_4d_axis2->grad = make_shared<Tensor>(
        vector<int>{2, 3, 2, 2},
        vector<float>{
            // Batch 0, Channel 0
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 0, Channel 1
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 0, Channel 2
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 0
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 1
            1.0f, 1.0f, 0.0f, 0.0f,
            // Batch 1, Channel 2
            1.0f, 1.0f, 0.0f, 0.0f
        },
        true
    );

    softmax_output_4d_axis2->backward();
    cout << "4D Gradients for SOFTMAX along axis 2:" << endl;
    softmax_input_4d->grad->print();

    // ===== SIMPLE 2x2 SOFTMAX TEST (Batch 0, Channel 0) =====
    cout << "\n" << string(50, '=') << endl;
    cout << "SIMPLE 2x2 SOFTMAX TEST (Batch 0, Channel 0)" << endl;
    cout << string(50, '=') << endl;

    // Extract batch 0, channel 0 from the 4D tensor (should be 2x2)
    // From the 4D input: [[[0.1, 0.2], [0.3, 0.4]], ...]
    // We want: [[0.1, 0.2], [0.3, 0.4]]
    auto simple_2x2 = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
        true
    );

    cout << "Simple 2x2 input tensor:" << endl;
    simple_2x2->print();
    // Expected: Tensor(2, 2): [0.1, 0.2], [0.3, 0.4]

    // Apply softmax along axis 1 (row-wise)
    auto simple_softmax = simple_2x2->softmax(1);
    cout << "Simple 2x2 SOFTMAX along axis 1 (row-wise):" << endl;
    simple_softmax->print();
    // Expected: Row 1: exp(0.1)=1.105, exp(0.2)=1.221, sum=2.326
    // Expected: Row 1: [0.475, 0.525]
    // Expected: Row 2: exp(0.3)=1.350, exp(0.4)=1.492, sum=2.842
    // Expected: Row 2: [0.475, 0.525]

    // Set gradient to same pattern as before
    simple_softmax->grad = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{1.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    simple_softmax->backward();
    cout << "Simple 2x2 Gradients for SOFTMAX:" << endl;
    simple_2x2->grad->print();

    return 0;
}



// SOFTMAX TEST - Input tensor:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// SOFTMAX along axis 1 (softmax of each row):
// Tensor(2, 3):
// [0.0900306 , 0.244728 , 0.665241 ], [0.0900306 , 0.244728 , 0.665241 ]
// Gradients for SOFTMAX along axis 1:
// Tensor(2, 3):
// [0.0819251 , -0.022033, -0.059892], [-0.022033, 0.184836 , -0.162803]
// SOFTMAX along axis 0 (softmax of each column):
// Tensor(2, 3):
// [0.0474259 , 0.0474259 , 0.0474259 ], [0.952574 , 0.952574 , 0.952574 ]
// Gradients for SOFTMAX along axis 0:
// Tensor(2, 3):
// [0.127102 , -0.022033, -0.105069], [-0.0672097, 0.184836 , -0.117627]

// Second SOFTMAX test - Input tensor with negative values:
// Tensor(2, 2):
// [-2, 2 ], [-1, 1 ]
// SOFTMAX along axis 1 (with negative values):
// Tensor(2, 2):
// [0.0179862 , 0.982014 ], [0.119203 , 0.880797 ]
// Gradients for SOFTMAX with negative values:
// Tensor(2, 2):
// [0.0176627 , -0.0176627], [-0.104994, 0.104994 ]

// Third SOFTMAX test - Input tensor with large values:
// Tensor(1, 3):
// [100 , 101 , 102 ]
// SOFTMAX with large values (should be numerically stable):
// Tensor(1, 3):
// [0.0900306 , 0.244728 , 0.665241 ]
// Gradients for SOFTMAX with large values:
// Tensor(1, 3):
// [0.0819251 , -0.022033, -0.059892]



// #include "tensor.h"
// #include <iostream>

// int main() {
//     // ===== CROSS ENTROPY FUNCTION TEST =====
    
//     // Create input logits (predictions)
//     auto logits = make_shared<Tensor>(
//         vector<int>{2, 3},
//         vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
//         true
//     );

//     // Create target labels (one-hot encoded)
//     auto targets = make_shared<Tensor>(
//         vector<int>{2, 3},
//         vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
//         false
//     );

//     cout << "CROSS ENTROPY TEST" << endl;
//     cout << "Input logits:" << endl;
//     logits->print();
//     // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

//     cout << "Target labels (one-hot):" << endl;
//     targets->print();
//     // Expected: Tensor(2, 3): [0, 0, 1], [0, 1, 0]

//     // Compute cross entropy loss
//     auto loss = logits->cross_entropy(targets, 1, true);
//     cout << "Cross entropy loss:" << endl;
//     loss->print();
//     // Expected: Should compute cross entropy loss for each sample

//     // Set gradient for backward pass
//     loss->grad = make_shared<Tensor>(
//         vector<int>{2, 1},
//         vector<float>{1.0f, 1.0f},
//         true
//     );

//     // Compute gradients
//     loss->backward();
//     cout << "Gradients for logits:" << endl;
//     logits->grad->print();
//     // Expected: Gradients showing how changes in loss affect logits

//     // ===== HIGHER DIMENSION CROSS ENTROPY TEST =====
//     cout << "\n" << string(50, '=') << endl;
//     cout << "HIGHER DIMENSION CROSS ENTROPY TEST" << endl;
//     cout << string(50, '=') << endl;

//     // Create 3D input logits (batch_size=2, sequence_length=3, num_classes=4)
//     auto logits_3d = make_shared<Tensor>(
//         vector<int>{2, 3, 4},
//         vector<float>{
//             // Batch 0, Sequence 0: [0.1, 0.2, 0.3, 0.4]
//             0.1f, 0.2f, 0.3f, 0.4f,
//             // Batch 0, Sequence 1: [0.5, 0.6, 0.7, 0.8]
//             0.5f, 0.6f, 0.7f, 0.8f,
//             // Batch 0, Sequence 2: [0.9, 1.0, 1.1, 1.2]
//             0.9f, 1.0f, 1.1f, 1.2f,
//             // Batch 1, Sequence 0: [1.3, 1.4, 1.5, 1.6]
//             1.3f, 1.4f, 1.5f, 1.6f,
//             // Batch 1, Sequence 1: [1.7, 1.8, 1.9, 2.0]
//             1.7f, 1.8f, 1.9f, 2.0f,
//             // Batch 1, Sequence 2: [2.1, 2.2, 2.3, 2.4]
//             2.1f, 2.2f, 2.3f, 2.4f
//         },
//         true
//     );

//     // Create 3D target labels (one-hot encoded)
//     auto targets_3d = make_shared<Tensor>(
//         vector<int>{2, 3, 4},
//         vector<float>{
//             // Batch 0, Sequence 0: class 2
//             0.0f, 0.0f, 1.0f, 0.0f,
//             // Batch 0, Sequence 1: class 0
//             1.0f, 0.0f, 0.0f, 0.0f,
//             // Batch 0, Sequence 2: class 3
//             0.0f, 0.0f, 0.0f, 1.0f,
//             // Batch 1, Sequence 0: class 1
//             0.0f, 1.0f, 0.0f, 0.0f,
//             // Batch 1, Sequence 1: class 2
//             0.0f, 0.0f, 1.0f, 0.0f,
//             // Batch 1, Sequence 2: class 0
//             1.0f, 0.0f, 0.0f, 0.0f
//         },
//         false
//     );

//     cout << "3D Input logits shape: [" << logits_3d->shape[0] << ", " 
//          << logits_3d->shape[1] << ", " << logits_3d->shape[2] << "]" << endl;
//     cout << "3D Input logits:" << endl;
//     logits_3d->print();

//     cout << "3D Target labels (one-hot):" << endl;
//     targets_3d->print();

//     // Compute cross entropy loss along the last axis (axis 2)
//     auto loss_3d = logits_3d->cross_entropy(targets_3d, 2, true);
//     cout << "3D Cross entropy loss:" << endl;
//     loss_3d->print();
//     // Expected: Should compute cross entropy loss for each sequence position

//     // Set gradient for backward pass
//     loss_3d->grad = make_shared<Tensor>(
//         vector<int>{2, 3, 1},
//         vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
//         true
//     );

//     // Compute gradients
//     loss_3d->backward();
//     cout << "3D Gradients for logits:" << endl;
//     logits_3d->grad->print();

//     // ===== 4D TENSOR TEST (BATCH, CHANNELS, HEIGHT, WIDTH) =====
//     cout << "\n" << string(50, '=') << endl;
//     cout << "4D TENSOR CROSS ENTROPY TEST" << endl;
//     cout << string(50, '=') << endl;

//     // Create 4D input logits (batch_size=2, channels=3, height=2, width=2)
//     auto logits_4d = make_shared<Tensor>(
//         vector<int>{2, 3, 2, 2},
//         vector<float>{
//             // Batch 0, Channel 0
//             0.1f, 0.2f, 0.3f, 0.4f,
//             // Batch 0, Channel 1
//             0.5f, 0.6f, 0.7f, 0.8f,
//             // Batch 0, Channel 2
//             0.9f, 1.0f, 1.1f, 1.2f,
//             // Batch 1, Channel 0
//             1.3f, 1.4f, 1.5f, 1.6f,
//             // Batch 1, Channel 1
//             1.7f, 1.8f, 1.9f, 2.0f,
//             // Batch 1, Channel 2
//             2.1f, 2.2f, 2.3f, 2.4f
//         },
//         true
//     );

//     // Create 4D target labels (one-hot encoded)
//     auto targets_4d = make_shared<Tensor>(
//         vector<int>{2, 3, 2, 2},
//         vector<float>{
//             // Batch 0, Channel 0
//             1.0f, 0.0f, 0.0f, 1.0f,
//             // Batch 0, Channel 1
//             0.0f, 1.0f, 0.0f, 0.0f,
//             // Batch 0, Channel 2
//             0.0f, 0.0f, 1.0f, 0.0f,
//             // Batch 1, Channel 0
//             0.0f, 1.0f, 0.0f, 0.0f,
//             // Batch 1, Channel 1
//             1.0f, 0.0f, 0.0f, 1.0f,
//             // Batch 1, Channel 2
//             0.0f, 0.0f, 1.0f, 0.0f
//         },
//         false
//     );

//     cout << "4D Input logits shape: [" << logits_4d->shape[0] << ", " 
//          << logits_4d->shape[1] << ", " << logits_4d->shape[2] << ", " 
//          << logits_4d->shape[3] << "]" << endl;
//     cout << "4D Input logits:" << endl;
//     logits_4d->print();

//     cout << "4D Target labels (one-hot):" << endl;
//     targets_4d->print();

//     // Compute cross entropy loss along the channel axis (axis 1)
//     auto loss_4d = logits_4d->cross_entropy(targets_4d, 1, true);
//     cout << "4D Cross entropy loss:" << endl;
//     loss_4d->print();

//     // Set gradient for backward pass
//     loss_4d->grad = make_shared<Tensor>(
//         vector<int>{2, 1, 2, 2},
//         vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
//         true
//     );

//     // Compute gradients
//     loss_4d->backward();
//     cout << "4D Gradients for logits:" << endl;
//     logits_4d->grad->print();

//     return 0;
// }