#define ll long long
using namespace std;

#include <math_lib/math_lib.h>
#include <math_lib/matrix.h>
#include <cmath>
#include <vector>
#include <limits>

using namespace math_lib;

void test_sigmoid() {
    assert(abs(sigmoid(0.0f) - 0.5f) < 1e-6);
    assert(sigmoid(10.0f) > 0.99f);
    assert(sigmoid(-10.0f) < 0.01f);

    std::vector<std::vector<float>> m1_vec = {{0.0f, 1.0f}, {-1.0f, 2.0f}};
    Matrix m1 = Matrix(m1_vec);
    std::vector<std::vector<float>> expected1_vec = {{0.5f, 0.7310585786f}, {0.2689414214f, 0.880797078f}};
    Matrix expected1 = Matrix(expected1_vec);
    Matrix result1 = sigmoid(m1);
    assert(result1.numRows() == expected1.numRows() && result1.numCols() == expected1.numCols());
    for (size_t i = 0; i < result1.numRows(); ++i) {
        for (size_t j = 0; j < result1.numCols(); ++j) {
            assert(abs(result1[i][j] - expected1[i][j]) < 1e-6);
        }
    }
    cout << "PASSED test_sigmoid" << endl;
}

void test_average_error() {
    std::vector<std::vector<float>> y_true1_vec = {{1.0f}, {2.0f}, {3.0f}};
    Matrix y_true1 = Matrix(y_true1_vec);
    std::vector<std::vector<float>> y_pred1_vec = {{1.1f}, {2.2f}, {2.8f}};
    Matrix y_pred1 = Matrix(y_pred1_vec);
    assert(abs(average_error(y_true1, y_pred1) - (0.1f + 0.2f + 0.2f) / 3.0f) < 1e-6);

    std::vector<std::vector<float>> y_true2_vec = {{0.0f}, {0.0f}};
    Matrix y_true2 = Matrix(y_true2_vec);
    std::vector<std::vector<float>> y_pred2_vec = {{0.0f}, {0.0f}};
    Matrix y_pred2 = Matrix(y_pred2_vec);
    assert(abs(average_error(y_true2, y_pred2) - 0.0f) < 1e-6);

    cout << "PASSED test_average_error" << endl;
}

void test_squared_euclidean_distance() {
    vector<float> v1 = {1.0f, 2.0f, 3.0f};
    vector<float> v2 = {1.1f, 2.2f, 2.8f};
    assert(abs(squared_euclidean_distance(v1, v2) - (0.01f + 0.04f + 0.04f)) < 1e-6);

    std::vector<std::vector<float>> m1_vec = {{1.0f}, {2.0f}};
    Matrix m1 = Matrix(m1_vec);
    std::vector<std::vector<float>> m2_vec = {{1.0f}, {2.0f}};
    Matrix m2 = Matrix(m2_vec);
    assert(abs(squared_euclidean_distance(m1, m2) - 0.0f) < 1e-6);

    cout << "PASSED test_squared_euclidean_distance" << endl;
}

void test_mean_std_dev() {
    vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    assert(abs(mean(v1) - 3.0f) < 1e-6);
    assert(abs(std_dev(v1) - sqrt(2.0f)) < 1e-6);

    vector<float> v2 = {0.0f, 0.0f, 0.0f};
    assert(abs(mean(v2) - 0.0f) < 1e-6);
    assert(abs(std_dev(v2) - 0.0f) < 1e-6);

    cout << "PASSED test_mean_std_dev" << endl;
}

void test_normalize() {
    std::vector<std::vector<float>> m1_vec = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix m1 = Matrix(m1_vec);
    std::vector<std::vector<float>> expected1_vec = {{-1.22474f, 0.0f, 1.22474f}, {-1.22474f, 0.0f, 1.22474f}};
    Matrix expected1 = Matrix(expected1_vec);
    Matrix result1 = normalize(m1, 0);
    assert(result1.numRows() == expected1.numRows() && result1.numCols() == expected1.numCols());
    for (size_t i = 0; i < result1.numRows(); ++i) {
        for (size_t j = 0; j < result1.numCols(); ++j) {
            assert(abs(result1[i][j] - expected1[i][j]) < 1e-5);
        }
    }

    // std::vector<std::vector<float>> m2_vec = {{1.0f}, {2.0f}};
    // Matrix m2 = Matrix(m2_vec);
    // bool assert_triggered_axis1 = false;
    // try {
    //     normalize(m2, 1);
    // } catch (const char* msg) {
    //     if (string(msg).find("Assertion failed") != string::npos) {
    //         assert_triggered_axis1 = true;
    //     }
    // }
    // assert(assert_triggered_axis1);

    // bool assert_triggered_axis_other = false;
    // try {
    //     normalize(m2, 2);
    // } catch (const char* msg) {
    //     if (string(msg).find("Assertion failed") != string::npos) {
    //         assert_triggered_axis_other = true;
    //     }
    // }
    // assert(assert_triggered_axis_other);

    cout << "PASSED test_normalize" << endl;
}

void test_softmax() {
    std::vector<std::vector<float>> m1_vec = {{1.0f, 2.0f, 3.0f}};
    Matrix m1 = Matrix(m1_vec);
    std::vector<std::vector<float>> expected1_vec = {{0.09003057f, 0.24472847f, 0.66524096f}};
    Matrix expected1 = Matrix(expected1_vec);
    Matrix result1 = softmax(m1, 0);
    assert(result1.numRows() == expected1.numRows() && result1.numCols() == expected1.numCols());
    for (size_t i = 0; i < result1.numRows(); ++i) {
        for (size_t j = 0; j < result1.numCols(); ++j) {
            assert(abs(result1[i][j] - expected1[i][j]) < 1e-6);
        }
    }

    // std::vector<std::vector<float>> m2_vec = {{1.0f}, {2.0f}};
    // Matrix m2 = Matrix(m2_vec);
    // bool assert_triggered_axis1 = false;
    // try {
    //     softmax(m2, 1);
    // } catch (const char* msg) {
    //     if (string(msg).find("Assertion failed") != string::npos) {
    //         assert_triggered_axis1 = true;
    //     }
    // }
    // assert(assert_triggered_axis1);

    // bool assert_triggered_axis_other = false;
    // try {
    //     softmax(m2, 2);
    // } catch (const char* msg) {
    //     if (string(msg).find("Assertion failed") != string::npos) {
    //         assert_triggered_axis_other = true;
    //     }
    // }
    // assert(assert_triggered_axis_other);

    cout << "PASSED test_softmax" << endl;
}

void test_positional_encoder() {
    Matrix input_embeddings1 = Matrix(10, 5);
    int d_model1 = 5;
    Matrix result1 = positional_encoder(input_embeddings1, d_model1);
    assert(result1.numRows() == 10 && result1.numCols() == 5);
    cout << "PASSED test_positional_encoder" << endl;
}

void test_attention() {
    std::vector<std::vector<float>> q_vec = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    Matrix Q = Matrix(q_vec);
    std::vector<std::vector<float>> k_vec = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    Matrix K = Matrix(k_vec);
    std::vector<std::vector<float>> v_vec = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix V = Matrix(v_vec);
    int d_k = 2;
    std::vector<std::vector<float>> expected1_vec = {{1.6607116f, 2.6607117f}, {2.3392884f, 3.3392883f}};
    Matrix expected1 = Matrix(expected1_vec);
    Matrix result1 = attention(Q, K, V, d_k, false);
    assert(result1.numRows() == expected1.numRows() && result1.numCols() == expected1.numCols());
    for (size_t i = 0; i < result1.numRows(); ++i) {
        for (size_t j = 0; j < result1.numCols(); ++j) {
            assert(abs(result1[i][j] - expected1[i][j]) < 5e-4);
        }
    }

    std::vector<std::vector<float>> q2_vec = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    Matrix Q2 = Matrix(q2_vec);
    std::vector<std::vector<float>> k2_vec = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    Matrix K2 = Matrix(k2_vec);
    std::vector<std::vector<float>> v2_vec = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix V2 = Matrix(v2_vec);
    int d_k2 = 2;
    std::vector<std::vector<float>> expected2_vec = {{1.0f, 2.0f}, {2.339833f, 3.339833f}};
    Matrix expected2 = Matrix(expected2_vec);
    Matrix result2 = attention(Q2, K2, V2, d_k2, true);
    assert(result2.numRows() == expected2.numRows() && result2.numCols() == expected2.numCols());
    for (size_t i = 0; i < result2.numRows(); ++i) {
        for (size_t j = 0; j < result2.numCols(); ++j) {
            assert(abs(result2[i][j] - expected2[i][j]) < 5e-4);
        }
    }

    cout << "PASSED test_attention" << endl;
}

void test_add_and_norm() {
    std::vector<std::vector<float>> a_vec = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix A = Matrix(a_vec);
    std::vector<std::vector<float>> b_vec = {{1.0f, 1.0f}, {1.0f, 1.0f}};
    Matrix B = Matrix(b_vec);
    std::vector<std::vector<float>> expected_vec = {{-1.0f, 1.0f}, {-1.0f, 1.0f}};
    Matrix expected = Matrix(expected_vec);
    Matrix result = add_and_norm(A, B);
    assert(result.numRows() == expected.numRows() && result.numCols() == expected.numCols());
    for (size_t i = 0; i < result.numRows(); ++i) {
        for (size_t j = 0; j < result.numCols(); ++j) {
            assert(abs(result[i][j] - expected[i][j]) < 5e-4);
        }
    }
    cout << "PASSED test_add_and_norm" << endl;
}

void test_max_scalar_matrix() {
    std::vector<std::vector<float>> m1_vec = {{-1.0f, 0.5f}, {2.0f, -3.0f}};
    Matrix m1 = Matrix(m1_vec);
    std::vector<std::vector<float>> expected1_vec = {{0.0f, 0.5f}, {2.0f, 0.0f}};
    Matrix expected1 = Matrix(expected1_vec);
    Matrix result1 = max(0.0f, m1);
    assert(result1.numRows() == expected1.numRows() && result1.numCols() == expected1.numCols());
    for (size_t i = 0; i < result1.numRows(); ++i) {
        for (size_t j = 0; j < result1.numCols(); ++j) {
            assert(abs(result1[i][j] - expected1[i][j]) < 5e-4);
        }
    }

    std::vector<std::vector<float>> expected2_vec = {{0.0f, 0.5f}, {2.0f, 0.0f}};
    Matrix expected2 = Matrix(expected2_vec);
    Matrix result2 = max(m1, 0.0f);
    assert(result2.numRows() == expected2.numRows() && result2.numCols() == expected2.numCols());
    for (size_t i = 0; i < result2.numRows(); ++i) {
        for (size_t j = 0; j < result2.numCols(); ++j) {
            assert(abs(result2[i][j] - expected2[i][j]) < 5e-4);
        }
    }
    cout << "PASSED test_max_scalar_matrix" << endl;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    test_sigmoid();
    test_average_error();
    test_squared_euclidean_distance();
    test_mean_std_dev();
    test_normalize();
    test_softmax();
    test_positional_encoder();
    test_attention();
    test_add_and_norm();
    test_max_scalar_matrix();

    return 0;
}