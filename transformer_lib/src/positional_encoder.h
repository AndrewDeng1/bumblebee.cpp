#ifndef POSITIONAL_ENCODER_H
#define POSITIONAL_ENCODER_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "../../math_lib/src/math_lib.h"

using namespace std;

Matrix positional_encoder(const Matrix input_embeddings, int d_model);

#endif // POSITIONAL_ENCODER_H