#include <math_lib/math_lib.h>
#include <math_lib/tensor.h>

using namespace std;

namespace math_lib {

// Sigmoid function implementation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Optional: Sigmoid function for vectors (element-wise application)
Matrix sigmoid(const Matrix& m) {
    Matrix ret = Matrix(m.numRows(), m.numCols());
    for (size_t i = 0; i < m.numRows(); i++) {
        for (size_t j = 0; j < m.numCols(); j++) {
            ret[i][j] = sigmoid(m[i][j]);
        }
    }
    return ret;
}

float average_error(const Matrix& y_true, const Matrix& y_pred) {

    assert(y_true.numRows() == y_pred.numRows()&&y_true.numCols()==1&&y_pred.numCols()==1&&"y_true and y_pred must both be vector shape (nx1) and have equal number of rows.");

    float error = 0;
    for (size_t i = 0; i < y_true.numRows(); i++) {
        error += abs(y_true[i][0] - y_pred[i][0]);
    }
    return error / y_true.numRows();
}

float squared_euclidean_distance(const vector<float>& v1, const vector<float> v2){
    return squared_euclidean_distance(Matrix(v1), Matrix(v2));
}

float squared_euclidean_distance(const Matrix& v1, const Matrix& v2){
    assert(v1.numCols() == 1 && v2.numCols() == 1 && v1.numRows() == v2.numRows() && "Both vectors must be column vectors of the same size.");

    float sm=0;
    for(size_t i=0; i<v1.numRows(); i++){
        sm += pow(v1[i][0]-v2[i][0], 2);
    }

    return sm;
}

float mean(const vector<float>& v){
    float sm=0.0;
    for(int i=0; i<v.size(); i++){
        sm+=v[i];
    }
    return sm/((float)v.size());
}

float std_dev(const vector<float>& v){
    float mu=mean(v);
    float sm=0.0;
    for(int i=0; i<v.size(); i++){
        sm+=powf(v[i]-mu, 2.0);
    }
    return sqrt(((float)1.0/v.size())*sm);
}


shared_ptr<Tensor> add(const shared_ptr<Tensor>& t1, const shared_ptr<Tensor>& t2) {
    // Handle broadcasting for vector-to-matrix addition
    Matrix ret;
    bool t1_is_vector = t1->data.numCols() == 1;
    bool t2_is_vector = t2->data.numCols() == 1;
    
    if (t1_is_vector && !t2_is_vector) {
        // Broadcast t1 (vector) to match t2 (matrix) dimensions
        ret = Matrix(t2->data.numRows(), t2->data.numCols());
        for (int i = 0; i < t2->data.numRows(); i++) {
            for (int j = 0; j < t2->data.numCols(); j++) {
                ret[i][j] = t1->data[i][0] + t2->data[i][j];
            }
        }
    } else if (!t1_is_vector && t2_is_vector) {
        // Broadcast t2 (vector) to match t1 (matrix) dimensions
        ret = Matrix(t1->data.numRows(), t1->data.numCols());
        for (int i = 0; i < t1->data.numRows(); i++) {
            for (int j = 0; j < t1->data.numCols(); j++) {
                ret[i][j] = t1->data[i][j] + t2->data[i][0];
            }
        }
    } else {
        // Regular matrix addition
        ret = t1->data + t2->data;
    }
    
    auto ret_tensor = make_shared<Tensor>(ret, true);
    ret_tensor->parents.push_back(t1);
    ret_tensor->parents.push_back(t2);
    
    ret_tensor->backward_fn = [t1, t2, ret_tensor, t1_is_vector, t2_is_vector]() {
        if (!t1->requires_grad && !t2->requires_grad) return;
        
        if (t1->requires_grad) {
            if (t1_is_vector) {
                // For vector input, sum gradients across columns
                for (int i = 0; i < t1->grad.numRows(); i++) {
                    float sum = 0;
                    for (int j = 0; j < ret_tensor->grad.numCols(); j++) {
                        sum += ret_tensor->grad[i][j];
                    }
                    t1->grad[i][0] += sum;
                }
            } else {
                t1->grad += ret_tensor->grad;
            }
        }
        
        if (t2->requires_grad) {
            if (t2_is_vector) {
                // For vector input, sum gradients across columns
                for (int i = 0; i < t2->grad.numRows(); i++) {
                    float sum = 0;
                    for (int j = 0; j < ret_tensor->grad.numCols(); j++) {
                        sum += ret_tensor->grad[i][j];
                    }
                    t2->grad[i][0] += sum;
                }
            } else {
                t2->grad += ret_tensor->grad;
            }
        }
    };
    
    return ret_tensor;
}


std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B) {
    assert(A->data.numCols() == B->data.numRows() && "matmul: incompatible dimensions");

    Matrix result = A->data*(B->data);  // Forward pass
    auto out = std::make_shared<Tensor>(result, true);
    out->parents = {A, B};

    // Define backward function
    out->backward_fn = [A, B, out]() {
        const Matrix& dL_dOut = out->grad;  // ∂L/∂C

        if (A->requires_grad) {
            A->grad += dL_dOut*(B->data.T());  // ∂L/∂A = dL/dC · Bᵗ
        }

        if (B->requires_grad) {
            B->grad += A->data.T()*(dL_dOut);  // ∂L/∂B = Aᵗ · dL/dC
        }
    };

    return out;
}


std::shared_ptr<Tensor> normalize(const std::shared_ptr<Tensor>& x, float epsilon) {
    int rows = x->data.numRows();
    int cols = x->data.numCols();
    Matrix out_data(rows, cols);
    Matrix mu(rows, 1);
    Matrix sigma(rows, 1);
    Matrix centered(rows, cols);

    // Forward pass: row-wise normalization
    for (int i = 0; i < rows; ++i) {
        float avg = mean(x->data[i]);
        float std = std_dev(x->data[i]);
        mu[i][0] = avg;
        sigma[i][0] = std;

        for (int j = 0; j < cols; ++j) {
            centered[i][j] = x->data[i][j] - avg;
            out_data[i][j] = centered[i][j] / (std + epsilon);
        }
    }

    auto out = std::make_shared<Tensor>(out_data, true);
    out->parents = {x};

    // Backward pass
    out->backward_fn = [x, out, centered, sigma, epsilon]() {
        if (!x->requires_grad) return;

        int rows = x->data.numRows();
        int cols = x->data.numCols();
        Matrix dL_dX(rows, cols);  // ∂L/∂X

        for (int i = 0; i < rows; ++i) {
            float std = sigma[i][0] + epsilon;
            float inv_std = 1.0f / std;
            float inv_std3 = 1.0f / (std * std * std);

            // Compute ∂L/∂x_i
            float sum_dy = 0.0f;
            float sum_dy_centered = 0.0f;

            for (int j = 0; j < cols; ++j) {
                float dy = out->grad[i][j];
                sum_dy += dy;
                sum_dy_centered += dy * centered[i][j];
            }

            for (int j = 0; j < cols; ++j) {
                float dy = out->grad[i][j];
                float dx = inv_std * (dy - sum_dy / cols - centered[i][j] * inv_std3 * sum_dy_centered);
                dL_dX[i][j] = dx;
            }
        }

        x->grad += dL_dX;
    };

    return out;
}


std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& x) {
    int rows = x->data.numRows();
    int cols = x->data.numCols();
    Matrix out_data(rows, cols);

    // Forward pass: apply ReLU
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out_data[i][j] = std::max(0.0f, x->data[i][j]);
        }
    }

    // Create output tensor
    auto out = std::make_shared<Tensor>(out_data, true);
    out->parents = {x};

    // Backward pass: ReLU derivative
    out->backward_fn = [x, out]() {
        if (!x->requires_grad) return;

        int rows = x->data.numRows();
        int cols = x->data.numCols();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float local_grad = (x->data[i][j] > 0.0f) ? 1.0f : 0.0f;
                x->grad[i][j] += out->grad[i][j] * local_grad;
            }
        }
    };

    return out;
}


std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& x, float epsilon) {
    int rows = x->data.numRows();
    int cols = x->data.numCols();
    Matrix out_data(rows, cols);

    // Compute row-wise softmax
    for (int i = 0; i < rows; ++i) {
        float max_val = x->data[i][0];
        for (int j = 1; j < cols; ++j) {
            if (x->data[i][j] > max_val)
                max_val = x->data[i][j];
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            out_data[i][j] = std::exp(x->data[i][j] - max_val);
            sum += out_data[i][j];
        }

        for (int j = 0; j < cols; ++j) {
            out_data[i][j] /= sum;
        }
    }

    auto out = std::make_shared<Tensor>(out_data, true);
    out->parents = {x};

    // Backward function using Jacobian-vector product trick
    out->backward_fn = [x, out]() {
        if (!x->requires_grad) return;

        int rows = x->data.numRows();
        int cols = x->data.numCols();
        Matrix dL_dX(rows, cols);

        for (int i = 0; i < rows; ++i) {
            // dot(out_grad_row, softmax_row)
            float dot = 0.0f;
            for (int j = 0; j < cols; ++j)
                dot += out->grad[i][j] * out->data[i][j];

            // compute final gradient
            for (int j = 0; j < cols; ++j) {
                float soft = out->data[i][j];
                float dy = out->grad[i][j];
                dL_dX[i][j] = soft * (dy - dot);
            }
        }

        x->grad += dL_dX;
    };

    return out;
}


shared_ptr<Tensor> add_and_norm(const shared_ptr<Tensor>& A, const shared_ptr<Tensor>& B){
    assert(A->data.numRows()==B->data.numRows()&&A->data.numCols()==B->data.numCols()&&"Add and norm operation requires A.numRows()==B.numRows()&&A.numCols()==B.numCols()");
    return normalize(add(A, B));
}


Matrix positional_encoder(const Matrix input_embeddings, int d_model) {
    Matrix ret = input_embeddings;
    
    for(int pos=0; pos<input_embeddings.numRows(); pos++){
        for(int i=0; i<input_embeddings.numCols(); i++){
            if(i%2==0){
                ret[pos][i]=sin(pos/powf(10000.0, (float)i/d_model));
            } else {
                ret[pos][i]=cos(pos/powf(10000.0, (float)(i-1)/d_model));
            }
        }
    }

    return ret;
}

std::shared_ptr<Tensor> attention(
    const std::shared_ptr<Tensor>& Q,
    const std::shared_ptr<Tensor>& K,
    const std::shared_ptr<Tensor>& V,
    int d_k,
    bool masked = false
) {
    // Step 1: Compute scores = (Q * Kᵀ) / sqrt(d_k)
    Matrix K_T = K->data.T();
    Matrix scores_data(Q->data.numRows(), K->data.numRows());

    for (int i = 0; i < Q->data.numRows(); ++i) {
        for (int j = 0; j < K_T.numCols(); ++j) {
            float dot = 0.0f;
            for (int k = 0; k < Q->data.numCols(); ++k) {
                dot += Q->data[i][k] * K_T[k][j];
            }
            scores_data[i][j] = dot / std::sqrt(static_cast<float>(d_k));
            if (masked && i < j) {
                scores_data[i][j] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    auto scores = std::make_shared<Tensor>(scores_data, true);
    auto softmax_scores = softmax(scores);  // Returns shared_ptr<Tensor>

    // Step 2: Output = softmax(scores) * V
    auto out = matmul(softmax_scores, V);

    // Step 3: Define backward function
    out->parents = {Q, K, V};  // Needed for context
    out->backward_fn = [Q, K, V, softmax_scores, scores, out, d_k]() {
        if (!Q->requires_grad && !K->requires_grad && !V->requires_grad) return;

        // Let:
        // A = softmax(QKᵀ / sqrt(d_k))
        // out = A * V
        // We already have: softmax_scores = A

        const Matrix& A = softmax_scores->data;
        const Matrix& dL_dOut = out->grad;

        // dL/dV = Aᵀ * dL/dOut
        if (V->requires_grad) {
            V->grad += softmax_scores->data.T()*dL_dOut;
        }

        // dL/dA = dL/dOut * Vᵀ
        Matrix dL_dA = dL_dOut*V->data.T();

        // dA/dscores is handled inside softmax.backward_fn
        softmax_scores->grad += dL_dA;

        // dL/dscores = softmax_backward_fn will be invoked when softmax_scores->backward_fn() is called

        // Now softmax_scores will push into scores->grad,
        // and we use that to compute gradients w.r.t. Q and K

        // We assume that softmax's backward_fn has already populated scores->grad at this point.
        // scores = QKᵀ / sqrt(d_k)
        // So:
        // dL/dQ = dL/dscores * K / sqrt(d_k)
        // dL/dK = (dL/dscores)ᵀ * Q / sqrt(d_k)

        if (Q->requires_grad || K->requires_grad) {
            Matrix dL_dScores = scores->grad;  // populated via softmax.backward_fn

            if (Q->requires_grad) {
                Q->grad += dL_dScores*K->data / std::sqrt(static_cast<float>(d_k));
            }

            if (K->requires_grad) {
                K->grad += dL_dScores.T()*Q->data / std::sqrt(static_cast<float>(d_k));
            }
        }
    };

    return out;
}

Matrix max(float n, const Matrix& m){
    Matrix ret = Matrix(m.numRows(), m.numCols());

    for(int i=0; i<m.numRows(); i++){
        for(int j=0; j<m.numCols(); j++){
            ret[i][j]=std::max(n, m[i][j]);  // make sure overloads properly
        }
    }

    return ret;
}

Matrix max(const Matrix& m, float n){
    return max(n, m);
}

void xavier_uniform_initialization(Matrix& m, int d_in, int d_out){
    float lower=-std::sqrt(6.0f/(float)(d_in+d_out));
    float upper=std::sqrt(6.0f/(float)(d_in+d_out));

    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> dist(lower, upper); // Uniform range [-bound, bound]

    // Fill the matrix with random values
    for (int i = 0; i < m.numRows(); ++i) {
        for (int j = 0; j < m.numCols(); ++j) {
            m[i][j] = dist(gen); // Sample and assign
        }
    }
}

shared_ptr<Tensor> concat(const vector<shared_ptr<Tensor>>& tensors, int axis) {
    assert(axis >= 0 && axis <= 1 && "Axis must be an integer between 0 and 1 inclusive.");
    assert(!tensors.empty() && "Cannot concatenate empty list of tensors");

    // Forward pass: concatenate the data matrices
    Matrix result = tensors[0]->data;
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = result.concat(tensors[i]->data, axis);
    }

    // Create output tensor
    auto out = make_shared<Tensor>(result, true);
    out->parents = tensors;

    // Backward pass
    out->backward_fn = [tensors, out, axis]() {
        if (axis == 0) {  // Concatenate along columns
            size_t col_offset = 0;
            for (const auto& tensor : tensors) {
                if (tensor->requires_grad) {
                    // Copy the corresponding columns from the gradient
                    for (size_t i = 0; i < tensor->data.numRows(); ++i) {
                        for (size_t j = 0; j < tensor->data.numCols(); ++j) {
                            tensor->grad[i][j] += out->grad[i][j + col_offset];
                        }
                    }
                }
                col_offset += tensor->data.numCols();
            }
        } else {  // Concatenate along rows
            size_t row_offset = 0;
            for (const auto& tensor : tensors) {
                if (tensor->requires_grad) {
                    // Copy the corresponding rows from the gradient
                    for (size_t i = 0; i < tensor->data.numRows(); ++i) {
                        for (size_t j = 0; j < tensor->data.numCols(); ++j) {
                            tensor->grad[i][j] += out->grad[i + row_offset][j];
                        }
                    }
                }
                row_offset += tensor->data.numRows();
            }
        }
    };

    return out;
}

shared_ptr<Tensor> embed(
    const vector<int>& token_indices,  // Sequence of token indices
    const shared_ptr<Tensor>& token_embeddings,  // Shape: (V, d_model)
    int d_model
) {
    // Create output tensor
    auto output = make_shared<Tensor>(Matrix(token_indices.size(), d_model), true);
    
    // For each token index in the sequence, look up its embedding
    for (int i = 0; i < token_indices.size(); ++i) {
        int token_idx = token_indices[i];
        // Copy the embedding for this token
        for (int j = 0; j < d_model; ++j) {
            output->data[i][j] = token_embeddings->data[token_idx][j];
        }
    }
    
    // Set up backward function to propagate gradients back to token embeddings
    output->backward_fn = [token_embeddings, token_indices, d_model, output]() {
        if (!token_embeddings->requires_grad) return;
        
        // For each token in the input sequence
        for (int i = 0; i < token_indices.size(); ++i) {
            int token_idx = token_indices[i];
            // Accumulate gradients for this token's embedding
            for (int j = 0; j < d_model; ++j) {
                token_embeddings->grad[token_idx][j] += output->grad[i][j];
            }
        }
    };
    
    // Set token_embeddings as a parent for gradient tracking
    output->parents = {token_embeddings};
    
    return output;
}

} // namespace math_lib
