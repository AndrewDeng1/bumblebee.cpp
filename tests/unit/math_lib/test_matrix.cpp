#include <math_lib/matrix.h>

using namespace std;

// void test_init(){
//     Matrix m=Matrix();
//     assert(m.numRows()==0&&m.numCols()==0);
    
//     Matrix a=Matrix(5, 10);
//     assert(a.numRows()==5&&a.numCols()==10);

//     cout<<"PASSED test_init"<<endl;
// }

// void test_operators(){

//     vector<vector<float>>arr1={{2, 4}, {6, 8}, {10, 12}};
//     Matrix exp=Matrix(arr1);
//     Matrix m=Matrix(arr1);
//     vector<vector<float>>ret;
//     for(int i=0; i<m.numRows(); i++){
//         ret.push_back(m[i]);
//     }

//     assert(exp==ret);


//     vector<float>arr2={-2, 2};
//     Matrix m2=m+arr2;
//     Matrix expected2 = Matrix({{0, 6}, {4, 10}, {8, 14}});

//     assert(m2==expected2);


//     Matrix m3=Matrix({{1, 2}, {3, 4}});
//     Matrix m4=Matrix({{-1, -1}, {-1, -1}});
//     Matrix m5=Matrix({{1, 2, 3}, {4, 5, 6}});
//     Matrix m_expected_add=Matrix({{0, 1}, {2, 3}});
//     Matrix m_expected_subtract=Matrix({{2, 3}, {4, 5}});
//     Matrix m_expected_prod=Matrix({{9, 12, 15}, {19, 26, 33}});
//     Matrix m_add=m3+m4;
//     Matrix m_subtract=m3-m4;
//     Matrix m_prod=m3*m5;

//     assert(m_expected_add==m_add);
//     assert(m_expected_subtract==m_subtract);
//     assert(m_expected_prod==m_prod);


//     Matrix m6=m3.concat(m4);
//     Matrix m_expected_concat=Matrix({{1, 2, -1, -1}, {3, 4, -1, -1}});

//     assert(m6==m_expected_concat);
//     assert((m3*0)==(0*m3)&&(m3*0)==Matrix({{0, 0}, {0, 0}}));
//     assert((m3/1.0)==m3);

//     cout<<"PASSED test_operators"<<endl;
// }


int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);

    // test_init();
    // test_operators();
}