#pragma once
#include "utils.h"

//十进制转二进制
void DecToBin(const int dec_num, vector<bool>& bin);

//二进制转十进制
void BinToDec(const vector<bool>& bin, int& dec_num);

//二进制转格雷码
void BinToGrayCode(const vector<bool>& bin, vector<bool>& grayCode);

//格雷码转二进制
void GrayCodeToBin(const vector<bool>& grayCode, vector<bool>& bin);