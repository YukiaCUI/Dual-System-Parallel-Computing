#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <winsock2.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <omp.h>
#include <xmmintrin.h> // SSE
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>  
#include <sstream> 

#pragma comment(lib, "Ws2_32.lib")

#define MAX_THREADS 64
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*�����ֵ����������*/
#undef max
const int BUFFER_SIZE = 1024;

//���������ݶ���Ϊ��
float rawFloatData[DATANUM];
float rawFloatData_sortresult_local[DATANUM];
float rawFloatData_sortresult_remote[DATANUM];
float rawFloatData_sortresult_combined[DATANUM * 2];


// �����߳����ݽṹ
struct ThreadData {
	const float* data;
	int start;
	int end;
	double result;
};


// ��������ͺ���
float sum_speedup(const float data[], int len) {
	double totalSum = 0.0;
	int numThreads = omp_get_max_threads();
	int chunkSize = len / numThreads;

	// OpenMP ��������
#pragma omp parallel reduction(+:totalSum)
	{
		int threadId = omp_get_thread_num();
		int start = threadId * chunkSize;
		int end = (threadId + 1) == numThreads ? len : (threadId + 1) * chunkSize;

		__m256 vdata, vsqrt;
		float logArray[8];

		for (int i = start; i < end; i += 8) {
			vdata = _mm256_loadu_ps(&data[i]); // ʹ�� SSE ���� 4 ��������
			vsqrt = _mm256_sqrt_ps(vdata); // ʹ�� SSE ִ�п�������
			vsqrt = _mm256_sqrt_ps(vsqrt); // ʹ�� SSE �ٴ�ִ�п������㣬��linux�ϱ���ͳһ
			_mm256_storeu_ps(logArray, vsqrt); // ������洢��������
			totalSum += logArray[0] + logArray[1] + logArray[2] + logArray[3]; // �ۼӽ��
		}
	}

	return totalSum;
}


//���������ֵ
float max_speedup(const float data[], int len) {
	double maxResult = -INFINITY;
	int numThreads = omp_get_max_threads();
	int chunkSize = len / numThreads;

	// OpenMP ��������
#pragma omp parallel
	{
		int threadId = omp_get_thread_num();
		int start = threadId * chunkSize;
		int end = (threadId + 1) == numThreads ? len : (threadId + 1) * chunkSize;

		double localMax = -INFINITY;
		__m256 vdata, vsqrt;
		float logArray[8];

		for (int i = start; i < end; i += 8) {
			vdata = _mm256_loadu_ps(&data[i]); // ʹ�� SSE ���� 4 ��������
			vsqrt = _mm256_sqrt_ps(vdata); // ʹ�� SSE ִ�п�������
			vsqrt = _mm256_sqrt_ps(vsqrt); // ʹ�� SSE �ٴ�ִ�п������㣬��linux�ϱ���ͳһ
			_mm256_storeu_ps(logArray, vsqrt); // ������洢��������

			for (int j = 0; j < 8; ++j) {
				if (logArray[j] > localMax) {
					localMax = logArray[j]; // ���¾ֲ����ֵ
				}
			}
		}

#pragma omp critical
		if (localMax > maxResult) {
			maxResult = localMax;
		}
	}
	return maxResult;
}

float sqrtA[DATANUM];
void normal_sort_speedup(float* A, int x, int y, float* T)
{
	if (y - x > 1) {
		int m = x + (y - x) / 2;
		int p = x, q = m, i = x;
#pragma omp task shared(A, T, sqrtA)
		normal_sort_speedup(A, x, m, T);

#pragma omp task shared(A, T, sqrtA)
		normal_sort_speedup(A, m, y, T);

		// �ȴ������������
#pragma omp taskwait

//������������sqrt��log�����ӻ�ʱ��
		__m256 vdata, vsqrt;
		float logArray[8];

		for (int i = x; i < y; i += 8) {
			__m256 vdata = _mm256_loadu_ps(&A[i]);
			vsqrt = _mm256_sqrt_ps(vdata); // ʹ�� SSE ִ�п�������
			vsqrt = _mm256_sqrt_ps(vsqrt); // ʹ�� SSE �ٴ�ִ�п������㣬��linux�ϱ���ͳһ
			_mm256_storeu_ps(logArray, vsqrt); // ������洢��������
		}

		while (p < m || q < y) {
			if (q >= y || (p < m && sqrtA[p] < sqrtA[q])) {
				T[i++] = A[p++];
			}
			else {
				T[i++] = A[q++];
			}
		}
		for (i = x; i < y; i++) {
			A[i] = T[i];
		}
	}
}

bool isSortTrue_speedup(float data[]) {
	bool sorted = true;

#pragma omp parallel for
	for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); i++) {
#pragma omp critical
		if (data[i] > data[i + 1])
			sorted = false;

	}
	return sorted;
}


float sort_speedup(const float data[], const int len, float result[])
{
#pragma omp parallel
	{
#pragma omp single nowait // ֻ��һ���߳���������������
		normal_sort_speedup(rawFloatData, 0, len - 1, result);
	}

	if (isSortTrue_speedup(result) == false)
		return 0;
	else
		return 1;
}





/*----------------------------�����ٺ�������------------------------------*/
//���������
float sum(const float data[], const int len)
{
	double result = 0.0;
	double c = 0.0;
	for (int i = 0; i < len; i++) {
		double corrected = log(sqrt(data[i])) - c;
		double newresult = result + corrected;
		c = (newresult - result) - corrected;
		result = newresult;
	}

	return static_cast<float>(result);
}

//�����������ֵ
float max(const float data[], const int len)
{
	double result = 0.0f;
	for (int i = 0; i < len; i++) {
		if (result <= log(sqrt(data[i])))
			result = log(sqrt(data[i]));
	}
	return result;
}

//�����������ж������Ƿ���ȷ
void normal_sort(float* A, int x, int y, float* T)
{
	if (y - x > 1) {
		int m = x + (y - x) / 2;
		int p = x, q = m, i = x;
		normal_sort(A, x, m, T);
		normal_sort(A, m, y, T);
		while (p < m || q < y) {
			if (q >= y || (p < m && log(sqrt(A[p])) <= log(sqrt(A[q])))) {
				T[i++] = A[p++];
			}
			else {
				T[i++] = A[q++];
			}
		}
		for (i = x; i < y; i++) {
			A[i] = T[i];
		}
	}
}

bool isSortTrue(float data[])
{
	for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); i++)
		if (data[i] > data[i + 1])
			return false;
	return true;
}

float sort(const float data[], const int len, float  result[])
{
	normal_sort(rawFloatData, 0, len - 1, result);
	if (isSortTrue(result) == false)
		return 0;
	else
		return 1;
}

void sendArray(int socket, float* data, const int len) {

	const int block_len = BUFFER_SIZE / sizeof(float);
	int send_len = 0;
	int send_size;
	while (send_len < len) {
		send_size = block_len < (len - send_len) ? block_len : len - send_len;
		size_t bytesSent = send(socket, (char*)(data + send_len), send_size * sizeof(float), 0);

		send_len += bytesSent / sizeof(float);
		//std::cout << "while ��" << "len=" << len << " recv_len=" << send_len << std::endl;
	}
}


int main() {
	WSADATA wsaData;
	SOCKET clientSocket;
	struct sockaddr_in server;

	// Initialize Winsock
	WSAStartup(MAKEWORD(2, 2), &wsaData);

	// Create socket
	clientSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (clientSocket == INVALID_SOCKET) {
		std::cerr << "Socket creation failed." << std::endl;
		return 1;
	}

	// Setup server structure
	server.sin_family = AF_INET;
	server.sin_addr.s_addr = inet_addr("192.168.43.147");
	server.sin_port = htons(8080);

	// Connect to server
	if (connect(clientSocket, (struct sockaddr*)&server, sizeof(server)) == SOCKET_ERROR) {
		std::cerr << "Connection failed." << std::endl;
		closesocket(clientSocket);
		return 1;
	}

	std::cout << "Connected to server." << std::endl;

	srand(time(NULL));
	LARGE_INTEGER start, end, frequency;
	int key, bag;
	float buffer[100] = { 0 };
	double interval;

	//���ݳ�ʼ��
	for (size_t i = 0; i < DATANUM; i++) {
		rawFloatData[i] = float(i + 1);
		rawFloatData_sortresult_local[i] = 0.0;
	}
	// ��������
	std::random_device rd;  // ���������
	std::mt19937 g(rd());   // Mersenne Twister �����������
	std::shuffle(rawFloatData, rawFloatData + DATANUM, g);


	/*key = -1;
	while (key!=0) {*/
	double floatsum = 0.0;
	double floatmax = 0.0;
	double floatsum_speedup = 0.0;
	double floatmax_speedup = 0.0;

	//recv(clientSocket, (char*)&key, sizeof(key), NULL);
	//std::cout << "��������Ϊ:" << key << std::endl;
	send(clientSocket, (char*)&bag, sizeof(bag), NULL);
	/*--------------------------�����ٴ��䲿��--------------------------------*/
	//��������ʹ���
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);//start
	floatsum = sum(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end);//end
	bag = floatsum;
	send(clientSocket, (char*)&bag, sizeof(bag), NULL);
	interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	std::cout << "1PC_sum: " << floatsum << std::endl;
	std::cout << "sum_time: " << interval << std::endl;



	//�����������ֵ����
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);//start
	floatmax = max(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end);//end
	interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	std::cout << "1PC_max: " << floatmax << std::endl;
	std::cout << "max_time: " << interval << std::endl;
	bag = floatmax;
	send(clientSocket, (char*)&bag, sizeof(bag), NULL);

	//������������
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);//start
	int flag = sort(rawFloatData, DATANUM, rawFloatData_sortresult_local);
	QueryPerformanceCounter(&end);//end
	interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	if (flag == 0)
		std::cout << "sort error" << std::endl;
	else
		std::cout << "sort success" << std::endl;
	//for (int i = 0; i < DATANUM; i++)
	//	rawFloatData_sortresult_remote[i] = rawFloatData_sortresult_local[i];
	std::cout << "sort_time: " << interval << std::endl;
	sendArray(clientSocket, rawFloatData_sortresult_local, DATANUM);



	/*---------------------------���ٴ��䲿��-----------------------------------*/
	//const char* sign = nullptr;
	//recv(clientSocket, (char*)&key, sizeof(key), NULL);
	//������ʹ���
	/*sign = "speedup_sum";
	send(clientSocket, sign, strlen(sign) + 1, 0);*/
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);//start
	floatsum_speedup = sum_speedup(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end);//end
	bag = floatsum_speedup;
	send(clientSocket, (char*)&bag, sizeof(bag), NULL);
	interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	std::cout << "1PC_sum_speedup: " << floatsum_speedup << std::endl;
	std::cout << "sum_time_speedup: " << interval << std::endl;




	//���������ֵ����
	/*sign = "speedup_max";
	send(clientSocket, sign, strlen(sign) + 1, 0);*/
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);//start
	floatmax_speedup = max_speedup(rawFloatData, DATANUM);
	QueryPerformanceCounter(&end);//end
	bag = floatmax_speedup;
	send(clientSocket, (char*)&bag, sizeof(bag), NULL);
	interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	std::cout << "1PC_max_speedup: " << floatmax_speedup << std::endl;
	std::cout << "max_speedup_time: " << interval << std::endl;



	//��������
	/*sign = "speedup_sort";
	send(clientSocket, sign, strlen(sign) + 1, 0);*/
	std::cout << "��ʼ����������" << std::endl;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);//start
	int flag_speedup = sort_speedup(rawFloatData, DATANUM, rawFloatData_sortresult_local);
	QueryPerformanceCounter(&end);//end
	interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	if (flag_speedup == 0)
		std::cout << "sort_speedup error" << std::endl;
	else
		std::cout << "sort_speedup success" << std::endl;
	std::cout << "sort_speedup_time: " << interval << std::endl;
	sendArray(clientSocket, rawFloatData_sortresult_local, DATANUM);

	// Close socket
	closesocket(clientSocket);
	WSACleanup();

	return 0;
}