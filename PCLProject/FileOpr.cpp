#include "FileOpr.h"

//读取CSV文件==============================================
bool ReadCSVFile(const string& filename, vector<vector<string>>& strArray)
{
	if (access(filename.c_str(), 0) == -1)
		return false;
	ifstream inFile(filename, ios::in);
	if (!inFile.is_open())
		return false;
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
	return true;
}
//=========================================================

//写入CSV文件==============================================
bool WriteCSVFile(vector<vector<string>>& strArray, const string &filename)
{
	if (access(filename.c_str(), 0) == 0)
	{
		remove(filename.c_str());
	}
	ofstream outfile(filename, ios::out | ios::app);
	if (outfile.is_open())
	{
		for (uint i = 0; i < strArray.size(); ++i)
		{
			for (int j = 0; j < strArray[i].size(); ++j)
			{
				outfile << strArray[i][j] << ",";
			}
			outfile << endl;
		}
		outfile.close();
		return true;
	}
	else
		return false;
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

//写入path文件=============================================
bool WritePath2BinFile(std::vector<double> &poseDatas, const string& filename)
{
	ofstream of(filename, ios_base::binary);
	if (!of)
	{
		return false;
	}
	int pNum = poseDatas.size();
	of.write((char*)&pNum, sizeof(int));
	for (size_t i = 0; i < pNum; i++)
	{
		of.write((char*)&poseDatas[i], 8);
	}
	of.close();
	return true;
}
//=========================================================