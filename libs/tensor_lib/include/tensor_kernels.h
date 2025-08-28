#pragma once
#include "tensor.h"

void launchAdd(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launchSubtract(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launchMultiply(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launchDivide(shared_ptr<Tensor> a, shared_ptr<Tensor> b, shared_ptr<Tensor> result);
void launchBroadcast(shared_ptr<Tensor> a, shared_ptr<Tensor> b, vector<int>& padded_shae, vector<int>& padded_strides, bool matmul);