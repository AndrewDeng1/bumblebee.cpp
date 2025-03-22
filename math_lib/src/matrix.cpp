#include "matrix.h"
#include <stdlib.h>
#include <time.h>

Matrix::Matrix(): m(vector<vector<float>>(0, vector<float>(0, 0))){}

Matrix::Matrix(size_t rows, size_t cols): m(vector<vector<float>>(rows, vector<float>(cols, 0))){}

Matrix::Matrix(const vector<vector<float>>arr): m(arr){}

Matrix::Matrix(const vector<float>arr): m(vector<vector<float>>(arr.size(), vector<float>(1, 0))){
    for(size_t i=0; i<arr.size(); i++){
        m[i][0]=arr[i];
    }
}

// "const" indicates method isn't allowed to modify member variables, and can only access other "const" methods
size_t Matrix::numRows() const {
    return m.size();
}

size_t Matrix::numCols() const {
    if(numRows()==0){
        return 0;
    }
    return m[0].size();
}

vector<float>& Matrix::operator[](size_t row){
    return m[row];
}

const vector<float>& Matrix::operator[](size_t row) const {
    return m[row];
}

Matrix Matrix::operator+(const Matrix& matrix) const {

    assert(numRows()==matrix.numRows()&&"Matrices must have same number of rows.");
    assert(numCols()==matrix.numCols()&&"Matrices must have same number of columns.");

    Matrix temp = Matrix(numRows(), numCols());
    for(int i=0; i<numRows(); i++){
        for(int j=0; j<numCols(); j++){
            temp[i][j]=m[i][j]+matrix[i][j];
        }
    }

    return temp;
}

Matrix Matrix::operator-(const Matrix& matrix) const {

    assert(numRows()==matrix.numRows()&&"Matrices must have same number of rows.");
    assert(numCols()==matrix.numCols()&&"Matrices must have same number of columns.");

    Matrix temp = Matrix(numRows(), numCols());
    for(int i=0; i<numRows(); i++){
        for(int j=0; j<numCols(); j++){
            temp[i][j]=m[i][j]-matrix[i][j];
        }
    }

    return temp;
}

Matrix Matrix::operator*(const Matrix& matrix) const {

    assert(numCols()==matrix.numRows()&&"A must have as many columns as B has rows.");

    Matrix temp = Matrix(numRows(), matrix.numCols());
    for(int i=0; i<numRows(); i++){
        for(int j=0; j<matrix.numCols(); j++){
            for(int x=0; x<numCols(); x++){
                temp[i][j]+=m[i][x]*matrix[x][j];
            }
        }
    }

    return temp;
}

float Matrix::dot(const Matrix& matrix) const {

    assert(numCols()==matrix.numRows()&&numRows()==1&&matrix.numCols()==1&&"Dot product only exists for two vector-shaped matrices of shape 1 x n and n x 1, respectively.");

    Matrix temp = Matrix(numRows(), matrix.numCols());
    for(int i=0; i<numRows(); i++){
        for(int j=0; j<matrix.numCols(); j++){
            for(int x=0; x<numCols(); x++){
                temp[i][j]+=m[i][x]*matrix[x][j];
            }
        }
    }

    return temp[0][0];
}

Matrix Matrix::operator*(float k) const {

    Matrix temp = Matrix(numRows(), numCols());
    for(int i=0; i<numRows(); i++){
        for(int j=0; j<numCols(); j++){
            temp[i][j]=m[i][j]*k;
        }
    }

    return temp;
}

Matrix operator*(float k, const Matrix& m) {

    Matrix temp = Matrix(m.numRows(), m.numCols());
    for(size_t i=0; i<m.numRows(); i++){
        for(size_t j=0; j<m.numCols(); j++){
            temp[i][j]=m[i][j]*k;
        }
    }

    return temp;
}

Matrix Matrix::slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const {

    assert(row_start<=row_end&&col_start<=col_end&&row_start>=0&&row_end<=numRows()&&col_start>=0&&col_end<=numCols()&&"Slicing out of bounds.");

    Matrix temp = Matrix(row_end-row_start, col_end-col_start);
    for(int i=row_start; i<row_end; i++){
        for(int j=col_start; j<col_end; j++){
            temp[i-row_start][j-col_start]=m[i][j];
        }
    }

    return temp;
}

Matrix Matrix::sample(size_t n, size_t axis) const {

    assert((axis==0||axis==1)&&"Axis must be an integer between 0 and 1 inclusive.");

    if(axis==0){

        assert(n<=numRows()&&"Number of rows to sample must be less than or equal to number of rows in matrix.");

        Matrix ret = Matrix(n, numCols());

        srand(time(NULL));

        vector<bool>seen(numRows(), false);

        size_t cnt=0;
        while(cnt<numRows()){
            int idx=rand()%numRows();
            if(!seen[idx]){
                seen[idx]=true;
                for(size_t col=0; col<numCols(); col++){
                    ret[cnt][col]=m[idx][col];
                }
                cnt++;
            }
        }

        return ret;
    } else {

        assert(n<=numCols()&&"Number of columns to sample must be less than or equal to number of columns in matrix.");

        Matrix ret = Matrix(numRows(), n);

        srand(time(NULL));

        vector<bool>seen(numCols(), false);

        size_t cnt=0;
        while(cnt<numCols()){
            int idx=rand()%numCols();
            if(!seen[idx]){
                seen[idx]=true;
                for(size_t i=0; i<numRows(); i++){
                    ret[i][cnt]=m[i][idx];
                }
                cnt++;
            }
        }

        return ret;
    }
}

Matrix Matrix::T() const {
    
    Matrix temp = Matrix(numCols(), numRows());

    for(int i=0; i<numRows(); i++){
        for(int j=0; j<numCols(); j++){
            temp[j][i]=m[i][j];
        }
    }

    return temp;
}

Matrix Matrix::concat(const Matrix& matrix, int axis) const {

    assert(axis>=0&&axis<=1&&"Axis must be an integer between 0 and 1 inclusive.");

    if(axis==0){
        assert(numRows()==matrix.numRows()&&"Both matrices must have same number of rows to be concatenable along axis 0.");

        Matrix temp = Matrix(numRows(), numCols()+matrix.numCols());
        for(size_t j=0; j<numCols(); j++){
            for(size_t i=0; i<numRows(); i++){
                temp[i][j]=m[i][j];
            }
        }

        for(size_t j=0; j<matrix.numCols(); j++){
            for(size_t i=0; i<matrix.numRows(); i++){
                temp[i][numCols()+j]=matrix[i][j];
            }
        }

        return temp;
    } else {  // axis=1
        // CAREFUL OF 0 COLUMNS CASE
        //  Due to matrix being represented as 2d vector, 
        //  - Possible for matrix to have non-zero rows but 0 columns
        //  - Impossible for matrix to have 0 rows but non-zero columns
        //  Thus we assume if a matrix has 0 columns, it has infinitely many columns
        //  E.g. If matrix has 10 rows and 0 columns, and being concatenated along axis=1
        //  with matrix with 100 rows and 10 columns, resulting matrix is just the ladder 
        //  matrix.
        //      BE CAREFUL IF THIS IS NOT THE DESIRED OUTCOME

        // printf("numCols, m.numCols, %d %d\n", numCols(), matrix.numCols());
        assert((numCols()==0||matrix.numCols()==0||numCols()==matrix.numCols())&&"Both matrices must have same number of columns to be concatenable along axis 1.");

        Matrix temp = Matrix(numRows()+matrix.numRows(), max(numCols(), matrix.numCols()));
        for(size_t i=0; i<numRows(); i++){
            for(size_t j=0; j<numCols(); j++){
                temp[i][j]=m[i][j];
            }
        }

        for(size_t i=0; i<matrix.numRows(); i++){
            for(size_t j=0; j<matrix.numCols(); j++){
                temp[numRows()+i][j]=matrix[i][j];
            }
        }

        return temp;
    }
    
}

Matrix Matrix::concat(const Matrix& matrix) const{
    return concat(matrix, 0);
}

// Time complexity: O(n!)
// Space complexity: O((n!)^2)
float Matrix::det() const {

    assert(numRows()==numCols()&&"Number of rows must be equal to number of columns for determinant to exist.");

    if(numRows()==1&&numCols()==1){
        return m[0][0];
    }

    if(numRows()==2&&numCols()==2){
        return m[0][0]*m[1][1]-m[0][1]*m[1][0];
    }

    float val=0;

    for(int j=0; j<numCols(); j++){
        Matrix temp=slice(1, numRows(), 0, j).concat(slice(1, numRows(), j+1, numCols()));
        val+=(j%2==1?-1:1)*m[0][j]*temp.det();
    }
    
    return val;
}

float Matrix::minor(size_t row, size_t col) const {
    Matrix top_left = slice(0, row, 0, col);
    Matrix top_right = slice(0, row, col+1, numCols());
    Matrix bottom_left = slice(row+1, numRows(), 0, col);
    Matrix bottom_right = slice(row+1, numRows(), col+1, numCols());
 
    return top_left.concat(top_right).concat(bottom_left.concat(bottom_right), 1).det();
}

float Matrix::cofactor(size_t row, size_t col) const{
    return pow(-1, row+col)*minor(row, col);
}

Matrix Matrix::adjoint() const{
    Matrix temp = Matrix(numRows(), numCols());

    for(int i=0; i<numRows(); i++){
        for(int j=0; j<numCols(); j++){
            temp[i][j]=cofactor(i, j);
        }
    }

    return temp.T();
}

Matrix Matrix::inverse() const{
    
    assert(numRows()==numCols()&&"Inverse only exists for square matrices.");


    float determinant = det();
    // printf("Determinant: %.2f\n", determinant);

    assert(determinant!=0.0&&"Determinant of matrix is 0, so inverse doesn't exist.");

    return (1.0/((float)determinant))*adjoint();
}

vector<float> Matrix::toVec() const {

    assert(numCols()==1&&"Matrix must be in vector form (n x 1) to convert to vector.");

    vector<float>ret;
    for(size_t i=0; i<numRows(); i++){
        ret.push_back(m[i][0]);
    }

    return ret;
}

void Matrix::display() const {
    for(int i=0; i<numRows(); i++){
        for(int j=0; j<numCols(); j++){
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
}


// void solution(){
//     // Matrix m = Matrix(2, 2);
//     // vector<vector<float>>arr1 = {{1, 2}, {3, 4}};
//     // vector<vector<float>>arr2 = {{1, 2}, {3, 5}};

//     // Matrix m2 = Matrix(arr1);
//     // Matrix m3 = Matrix(arr2);

//     // m.display();
//     // m2.display();
//     // m3.display();

//     // (m+m2).display();
//     // (m2+m3).display();
//     // (m-m2).display();
//     // (m2-m3).display();
//     // (m*m2).display();
//     // (m2*m3).display();

//     // vector<vector<float>>arr = {
//     //     {1, -2, 3},
//     //     {2, 0, 3},
//     //     {1, 5, 4},
//     // };
//     // vector<vector<float>>arr = {
//     //     {1, 2, 0},
//     //     {3, -1, 2},
//     //     {-2, 3, -2},
//     // };
//     // vector<vector<float>>arr = {
//     //     {5, 6, 6, 8},
//     //     {2, 2, 2, 8},
//     //     {6, 6, 2, 8},
//     //     {2, 3, 6, 7}
//     // };
//     vector<vector<float>>arr = {
//         {1, 0, 4, 6},
//         {2, 5, 0, 3},
//         {-1, 2, 3, 5},
//         {2, 1, -2, 3}
//     };
//     Matrix m = Matrix(arr);

//     m.T().display();

//     Matrix temp=m.inverse();
//     printf("inverse of m is: \n");
//     temp.display();
//     printf("following should be identity matrix: \n");
//     (m*temp).display();
// }



// int main(){
//     ios::sync_with_stdio(0);
//     cin.tie(0);

//     solution();
// }