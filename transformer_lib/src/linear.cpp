#include "linear.h"

Linear::Linear(int d_model, int V) {
    this->d_model=d_model;
    this->V=V;
    
    this->W=Matrix(d_model, V);
    this->b=vector<float>(V);
}

Matrix Linear::forward(Matrix& X) const {
    return X*W+b;
}