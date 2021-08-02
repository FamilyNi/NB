#include "FileOperator.h"

//读取CSV文件==============================================
bool ReadCSVFile(const string& filename, vector<vector<string>>& strArray)
{
	if (access(filename.c_str(), 0) == -1)
		return -1;
	ifstream inFile(filename, ios::in);
	if (!inFile.is_open())
		return -1;
	string lineStr;
	if (strArray.size() != 0)
		strArray.resize(0);
	while (getline(inFile, lineStr))
	{
		stringstream ss(lineStr);
		string str;
		vector<string> lineArray;
		while (getline(ss, str, ','))
		{
			lineArray.push_back(str);
		}
		strArray.push_back(lineArray);
	}
	inFile.close();
	return 0;
}
//=========================================================

//读取标定文件=============================================
bool readBinFile(const string& filename, vector<double>& data)
{
	if (access(filename.c_str(), 0) == -1)
		return false;
	ifstream inFile(filename, ios::binary | ios::in);
	if (!inFile.is_open())
		return false;
	inFile.seekg(0, ios::end);       //将指针只想末尾
	size_t length = inFile.tellg();  //获取文件长度
	if (length != sizeof(double) * 12)
		return false;
	if (data.size() != 12)
		data.resize(12);
	inFile.seekg(0, ios::beg);
	inFile.read((char*)& data[0], sizeof(double) * 12);
	inFile.close();
	return true;
}
//=========================================================