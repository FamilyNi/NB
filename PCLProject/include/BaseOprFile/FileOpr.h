#pragma once
#include "utils.h"

//读取CSV文件
bool ReadCSVFile(const string& filename, vector<vector<string>>& strArray);

//写入CSV文件
bool WriteCSVFile(vector<vector<string>>& strArray, const string &filename);

//读取标定文件
bool readBinFile(const string& filename, vector<double>& data);
