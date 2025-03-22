#ifndef DATA_FRAME_H
#define DATA_FRAME_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

class DataFrame {
    
    public:

        // Declare signature of constructor methods
        // * Favors use of column names over row names
        DataFrame(size_t rows, size_t cols);
        DataFrame(size_t rows, const vector<string> columns);
        DataFrame(const vector<vector<string>> arr);
        DataFrame(const vector<vector<float>> arr);
        DataFrame(const vector<vector<string>> arr, const vector<string> columns);
        DataFrame(const vector<vector<float>> arr, const vector<string> columns);
        DataFrame(const Matrix m);
        DataFrame(const Matrix m, const vector<string> columns);
        // DataFrame(const vector<float>& arr);

        // Gets number of rows/cols of DataFrame
        vector<size_t> getShape() const;

        // Used to index into the DataFrame via column name
        vector<string>& operator[](string column);
        const vector<string>& operator[](string column) const;
        vector<string>& operator[](size_t col_ind);
        const vector<string>& operator[](size_t col_ind) const;

        // Transpose
        DataFrame T() const;

        // Concatenates data frame horizontally
        DataFrame concat(const DataFrame& matrix, int axis) const;
        DataFrame concat(const DataFrame& matrix) const;

        // Returns a slice of the data frame
        DataFrame slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const;

        // Displays the data frame to console
        void display() const;
    
    private:
        
        // 2D dynamic arrays to represent data frame
        
        // ith row is ith column/column corresponding to ith column name in data frame
        // jth column is jth row/row corresponding to jth row name in data frame
        vector<vector<string>>data;
        vector<string>columns;
        vector<string>rows;
        map<string, size_t>column_to_index;
        map<string, size_t>row_to_index;

        vector<vector<string>>getData() const;
        void setData(const vector<vector<string>> data);

        void initColumns();
        void initColumns(const vector<string> columns);
        vector<string> getColumns() const;
        void setColumns(const vector<string> columns);

        void initRows();
        vector<string> getRows() const;
        void setRows(const vector<string> rows);

        map<string, size_t> getColumnToIndex() const;
        void setColumnToIndex(const map<string, size_t> column_to_index);

        map<string, size_t> getRowToIndex() const;
        void setRowToIndex(const map<string, size_t> row_to_index);
};

#endif // DATA_FRAME_H