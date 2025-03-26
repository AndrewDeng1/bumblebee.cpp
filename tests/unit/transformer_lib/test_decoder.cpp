// #include<bits/stdc++.h>
#define ll long long
using namespace std;

#include <math_lib/matrix.h>
// #include "math_lib/src/matrix.h"

void test_init(){
    Matrix m=Matrix();
    assert(m.numRows()==0&&m.numCols()==0);
    
    Matrix a=Matrix(5, 10);
    assert(a.numRows()==5&&a.numCols()==10);

    cout<<"PASSED test_init"<<endl;
}

void test_operators(){
    
    // vector<float>row1={1, 2};
    // vector<float>row2={4, 6};
    // vector<float>row3={8, 10};

    // vector<vector<float>>arr1={row1, row2, row3};

    // Matrix m=Matrix(arr1);
    // cout<<"PASSED test_operators"<<endl;
}


int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);

    test_init();
    test_operators();
}