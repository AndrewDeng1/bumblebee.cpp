#include "positional_encoder.h"
// #include "..\..\math_lib\src\math_lib.h"
// #include <stdlib.h>
// #include <cmath>

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