#include "data_frame.h"
#include "matrix.h"
#include <stdlib.h>
#include <time.h>

DataFrame::DataFrame(size_t rows, size_t cols): m(vector<vector<float>>(rows, vector<float>(cols, 0))){}

DataFrame::DataFrame(const vector<vector<float>>arr): m(arr){}

DataFrame::DataFrame(const vector<float>arr): m(vector<vector<float>>(arr.size(), vector<float>(1, 0))){
    for(size_t i=0; i<arr.size(); i++){
        m[i][0]=arr[i];
    }
}

vector<vector<string>> DataFrame::getData() const {
    return data;
}

void DataFrame::setData(vector<vector<string>> data) {
    this->data=data;
}

void DataFrame::initColumns() {
    vector<string>tmp=vector<string>();
    for(size_t i=0; i<getShape()[1]; i++){
        tmp.push_back(to_string(i));
    }
    setColumns(tmp);
}

void DataFrame::initColumns(const vector<string> columns) {
    this->columns=columns;
    for(size_t i=0; i<columns.size(); i++){
        getColumnToIndex()[columns[i]]=i;
    }
}

vector<string> DataFrame::getColumns() const {
    return columns;
}

void DataFrame::setColumns(const vector<string> columns) {
    this->columns=columns;
    setColumnToIndex(getColumns());
}

void DataFrame::initRows() {
    vector<string>tmp=vector<string>();
    for(size_t i=0; i<getShape()[0]; i++){
        tmp.push_back(to_string(i));
    }
    setRows(tmp);
    for(size_t i=0; i<getShape()[0]; i++){
        getRowToIndex()[getRows()[i]]=i;
    }
}

vector<string> DataFrame::getRows() const {
    return rows;
}

void DataFrame::setRows(const vector<string> rows) {
    this->rows=rows;
    setRowToIndex(getRows());
}

map<string, size_t> DataFrame::getColumnToIndex() const {
    return column_to_index;
}

void DataFrame::setColumnToIndex(const map<string, size_t> column_to_index) {
    this->column_to_index=column_to_index;
}

void DataFrame::setColumnToIndex(const vector<string> columns) {

    map<string, size_t>tmp;
    for(size_t i=0; i<columns.size(); i++){
        tmp[columns[i]]=i;
    }

    setColumnToIndex(tmp);
}

map<string, size_t> DataFrame::getRowToIndex() const {
    return row_to_index;
}

void DataFrame::setRowToIndex(const map<string, size_t> row_to_index) {
    this->row_to_index=row_to_index;
}

void DataFrame::setRowToIndex(const vector<string> rows) {

    map<string, size_t>tmp;
    for(size_t i=0; i<rows.size(); i++){
        tmp[rows[i]]=i;
    }

    setRowToIndex(tmp);
}

vector<size_t> DataFrame::getShape() const {
    return vector<size_t>{getData()[0].size(), getData().size()};
}

DataFrame::DataFrame(size_t rows, size_t cols) {
    setData(vector<vector<string>>(rows, vector<string>(cols, "")));
    initColumns();
    initRows();
}

DataFrame::DataFrame(size_t rows, const vector<string> columns) {
    setData(vector<vector<string>>(rows, vector<string>(columns.size(), "")));
    initColumns(columns);
    initRows();
}

DataFrame::DataFrame(const vector<vector<string>> arr) {
    vector<vector<string>>tmp=vector<vector<string>>(arr[0].size(), vector<string>(arr.size(), ""));
    for(size_t i=0; i<arr[0].size(); i++){
        for(size_t j=0; j<arr.size(); j++){
            tmp[i][j]=arr[j][i];
        }
    }
    initColumns();
    initRows();
}

DataFrame::DataFrame(const vector<vector<float>> arr){
    setData(vector<vector<string>>(arr[0].size(), vector<string>(arr.size(), "")));
    for(size_t i=0; i<arr[0].size(); i++){
        for(size_t j=0; j<arr.size(); j++){
            getData()[i][j]=to_string(arr[j][i]);
        }
    }
    initColumns();
    initRows();
}

DataFrame::DataFrame(const vector<vector<string>> arr, const vector<string> columns){
    vector<vector<string>>tmp=vector<vector<string>>(arr[0].size(), vector<string>(arr.size(), ""));
    for(size_t i=0; i<arr[0].size(); i++){
        for(size_t j=0; j<arr.size(); j++){
            tmp[i][j]=arr[j][i];
        }
    }
    initColumns(columns);
    initRows();
}

DataFrame::DataFrame(const vector<vector<float>> arr, const vector<string> columns){
    setData(vector<vector<string>>(arr[0].size(), vector<string>(arr.size(), "")));
    for(size_t i=0; i<arr[0].size(); i++){
        for(size_t j=0; j<arr.size(); j++){
            getData()[i][j]=to_string(arr[j][i]);
        }
    }
    initColumns(columns);
    initRows();
}

DataFrame::DataFrame(const Matrix m){
    vector<vector<string>>tmp=vector<vector<string>>(m.getCols(), vector<string>(m.getRows()));
    for(size_t i=0; i<m.getCols(); i++){
        for(size_t j=0; j<m.getRows(); j++){
            tmp[i][j]=tmp[j][i];
        }
    }
    setData(tmp);
    initColumns();
    initRows();
}

DataFrame::DataFrame(const Matrix m, const vector<string> columns){
    vector<vector<string>>tmp=vector<vector<string>>(m.getCols(), vector<string>(m.getRows()));
    for(size_t i=0; i<m.getCols(); i++){
        for(size_t j=0; j<m.getRows(); j++){
            tmp[i][j]=tmp[j][i];
        }
    }
    setData(tmp);
    initColumns(columns);
    initRows();
}

vector<string>& DataFrame::operator[](string column){
    return getData()[getColumnToIndex()[column]];
}

const vector<string>& DataFrame::operator[](string column) const {
    return getData()[getColumnToIndex()[column]];
}

vector<string>& DataFrame::operator[](size_t col_ind){
    return getData()[col_ind];
}

const vector<string>& DataFrame::operator[](size_t col_ind) const {
    return getData()[col_ind];
}

DataFrame DataFrame::T() const {

    vector<vector<string>>tmp=vector<vector<string>>(getShape()[1], vector<string>(getShape()[0]));
    for(size_t i=0; i<getShape()[1]; i++){
        for(size_t j=0; j<getShape()[0]; j++){
            tmp[i][j]=getData()[j][i];
        }
    }

    DataFrame df = DataFrame(tmp);
    df.setColumns(getRows());
    df.setRows(getColumns());

    return df;
}

DataFrame DataFrame::concat(const DataFrame& data_frame, int axis) const {

    assert(axis>=0&&axis<=1&&"Axis must be an integer between 0 and 1 inclusive.");

    if(axis==0){
        assert(getShape()[0]==data_frame.getShape()[0]&&"Both matrices must have same number of rows to be concatenable along axis 0.");

        DataFrame temp = DataFrame(getShape()[0], getShape[1]+data_frame.getShape()[1]);
        for(size_t j=0; j<getShape()[1]; j++){
            for(size_t i=0; i<getShape()[0]; i++){
                temp[i][j]=getData()[i][j];
            }
        }

        for(size_t j=0; j<data_frame.getShape()[1]; j++){
            for(size_t i=0; i<data_frame.getShape()[0]; i++){
                temp[i][getShape()[1]+j]=data_frame[i][j];
            }
        }

        return temp;
    } else {  // axis=1
        // CAREFUL OF 0 COLUMNS CASE
        //  Due to data_frame being represented as 2d vector, 
        //  - Possible for data_frame to have non-zero rows but 0 columns
        //  - Impossible for data_frame to have 0 rows but non-zero columns
        //  Thus we assume if a data_frame has 0 columns, it has infinitely many columns
        //  E.g. If data_frame has 10 rows and 0 columns, and being concatenated along axis=1
        //  with data_frame with 100 rows and 10 columns, resulting data_frame is just the ladder 
        //  data_frame.
        //      BE CAREFUL IF THIS IS NOT THE DESIRED OUTCOME

        // printf("getCols, m.getCols, %d %d\n", getCols(), data_frame.getCols());
        // assert((getCols()==0||data_frame.getCols()==0||getCols()==data_frame.getCols())&&"Both matrices must have same number of columns to be concatenable along axis 1.");

        // DataFrame temp = DataFrame(getRows()+data_frame.getRows(), max(getCols(), data_frame.getCols()));
        // for(size_t i=0; i<getRows(); i++){
        //     for(size_t j=0; j<getCols(); j++){
        //         temp[i][j]=m[i][j];
        //     }
        // }

        // for(size_t i=0; i<data_frame.getRows(); i++){
        //     for(size_t j=0; j<data_frame.getCols(); j++){
        //         temp[getRows()+i][j]=data_frame[i][j];
        //     }
        // }

        // return temp;
    }
    
}

DataFrame DataFrame::slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const {

    assert(row_start<=row_end&&col_start<=col_end&&row_start>=0&&row_end<=getRows()&&col_start>=0&&col_end<=getCols()&&"Slicing out of bounds.");

    DataFrame df = DataFrame(row_end-row_start, col_end-col_start);
    for(int i=row_start; i<row_end; i++){
        for(int j=col_start; j<col_end; j++){
            df[j-col_start][i-row_start]=getData()[j][i];
        }
    }

    vector<string>new_cols=vector<string>();
    for(size_t i=col_start; i<col_end; i++){
        new_cols.push_back(getColumns()[i]);
    }
    df.setColumns(new_cols);
    
    vector<string>new_rows=vector<string>();
    for(size_t i=row_start; i<row_end; i++){
        new_rows.push_back(getRows()[i]);
    }
    df.setRows(new_rows);

    return df;
}

DataFrame DataFrame::concat(const DataFrame& data_frame) const{
    return concat(data_frame, 0);
}

void DataFrame::display() const {
    cout<<" "*getRows()[0].length()+" ";
    for(int i=0; i<getColumns().size(); i++){
        cout<<getColumns()[i]<<" ";
    }
    cout<<endl;
    for(int i=0; i<getShape()[1]; i++){
        cout<<getRows()[i]<<" ";
        for(int j=0; j<getShape()[0]; j++){
            cout<<getData()[i][j]<<" ";
        }
        cout<<endl;
    }
}