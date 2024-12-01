#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <pthread.h>
#include <thread>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <algorithm>
#include <random>
#include <omp.h> 
#include <iomanip>

#define MAX_THREADS 64
#define SUBDATANUM 1000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*这个数值是总数据量*/
#undef max
const int BUFFER_SIZE = 1024;


//待测试数据定义为：
float rawFloatData[DATANUM];
float rawFloatData_sortresult_local[DATANUM];
float rawFloatData_sortresult_remote[DATANUM];
float rawFloatData_sortresult_combined[DATANUM * 2];


// 主加速求和函数
float sum_speedup(const float data[], int len) {
    double totalSum = 0.0;
    int numThreads = omp_get_max_threads();//获得最大线程数
    int chunkSize = len / numThreads;//分块，每个线程分工

    // OpenMP 并行区域
    #pragma omp parallel reduction(+:totalSum)//规约操作，将每个线程的值累加到原始的totalSum变量中
    {
        int threadId = omp_get_thread_num();//线程ID
        int start = threadId * chunkSize;
        int end = (threadId + 1) == numThreads ? len : (threadId + 1) * chunkSize;
        
        __m256 vdata, vroot;
        float rootArray[8];

        //每次循环处理4个浮点数，SSE指令操作4个单精度浮点数
        for (int i = start; i < end; i += 8) {
            vdata = _mm256_loadu_ps(&data[i]); // 使用 SSE 加载 8 个浮点数到256位sse寄存器内，加载的数据不需要在内存中对齐
            vroot = _mm256_sqrt_ps(vdata); // 并行计算平方根
            vroot = _mm256_sqrt_ps(vroot); // 再进行一次平方根元算，代替log运算（linux的gcc和clang里没有该函数）
            _mm256_storeu_ps(rootArray, vroot); // 将结果存储到数组中

            for (int j = 0; j < 4; ++j) {
                totalSum += rootArray[j]; 
            }
        }
    }
    return totalSum;
}


float max_speedup(const float data[], int len) {
    double maxResult = -INFINITY;
    int numThreads = omp_get_max_threads();
    int chunkSize = len / numThreads;

    // OpenMP 并行区域
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int start = threadId * chunkSize;
        int end = (threadId + 1) == numThreads ? len : (threadId + 1) * chunkSize;

        double localMax = -INFINITY;
        __m256 vdata, vroot;
        float rootArray[8];
        double logVal;

        for (int i = start; i < end; i += 8) {
            vdata = _mm256_loadu_ps(&data[i]); // 使用 SSE 加载 8 个浮点数
            vroot = _mm256_sqrt_ps(vdata); // 计算平方根
            vroot = _mm256_sqrt_ps(vroot); // 再进行一次平方根元算，代替log运算（linux的gcc和clang里没有该函数）
            _mm256_storeu_ps(rootArray, vroot); // 将结果存储到数组中

            for (int j = 0; j < 4; ++j) {
                logVal = rootArray[j];
                if (logVal > localMax) {
                    localMax = logVal; // 更新局部最大值
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
    //并行创建两个项目，分别处理数组两半
#pragma omp task shared(A, T, sqrtA) 
		normal_sort_speedup(A, x, m, T);

#pragma omp task shared(A, T, sqrtA)
		normal_sort_speedup(A, m, y, T);

#pragma omp taskwait//等待之前创建的所有任务完成


		for (int i = x; i < y; i += 8) {
			__m256 vdata = _mm256_loadu_ps(&A[i]);
			__m256 vsqrt = _mm256_sqrt_ps(vdata);
            vsqrt = _mm256_sqrt_ps(vsqrt); // 再进行一次平方根元算，代替log运算（linux的gcc和clang里没有该函数）
			_mm256_storeu_ps(&sqrtA[i], vsqrt);
		}

		while (p < m || q < y) {
			if (q >= y || (p < m && A[p] < A[q])) {
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
	for (size_t i = 0; i < sizeof(data)/sizeof(data[0]); i++) {
#pragma omp critical//保在任何给定时间只有一个线程可以执行
		if (data[i] > data[i + 1])
			sorted = false;

	}
	return sorted;
}



float sort_speedup(const float data[], const int len, float result[]) 
{
	normal_sort_speedup(rawFloatData, 0, len-1, result);

	if (isSortTrue_speedup(result) == false)
		return 0;
	else
		return 1;
}

void SortResultCombine_speedup(float* StrA, int lenA, float* StrB, int lenB, float* StrC) {
    // 并行区域
    #pragma omp parallel
    {
        // 获取线程数和当前线程ID
        int numThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();

        // 计算每个线程要处理的数据量
        int chunkSizeA = lenA / numThreads;
        int chunkSizeB = lenB / numThreads;

        // 计算每个线程处理的数据段的起始和结束位置
        int startA = chunkSizeA * threadId;
        int endA = (threadId == numThreads - 1) ? lenA : startA + chunkSizeA;

        int startB = chunkSizeB * threadId;
        int endB = (threadId == numThreads - 1) ? lenB : startB + chunkSizeB;

        int startC = startA + startB;
        int i = startA, j = startB, k = startC;

        // 合并两个数组的部分
        while (i < endA && j < endB) {
            if (StrA[i] < StrB[j]) {
                StrC[k++] = StrA[i++];
            } else {
                StrC[k++] = StrB[j++];
            }
        }

        // 处理剩余的元素
        while (i < endA) {
            StrC[k++] = StrA[i++];
        }
        while (j < endB) {
            StrC[k++] = StrB[j++];
        }
    }
}

/*-------------------------------------不加速部分相关函数--------------------------------------------------*/
//不加速求和
float sum(const float data[], const int len) 
{
	double result = 0.0;
    double c = 0.0;
    for (int i = 0; i < len; i++) {
        double corrected = log(sqrt(data[i])) - c;    // 当前值和之前累积的误差的差
        double newresult = result + corrected; // 将修正后的值加到总和中
        c = (newresult - result) - corrected;  // 计算并保存新的小误差
        result = newresult;
    }

    return static_cast<float>(result);
}

//不加速求最大值
float max(const float data[], const int len)
{
	double result = 0.0f;
	for (int i = 0; i < len; i++){
		if (result <= log(sqrt(data[i])))
			result = log(sqrt(data[i]));
	}
	return result;
}

//不加速排序、判断排序是否正确
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

bool isSortTrue(float data[]) {
	for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); i++)
		if (data[i] > data[i + 1])
			return false;
	return true;
}

float sort(const float data[], const int len, float result[]) 
{
	normal_sort(rawFloatData, 0, len - 1, result);
    //size_t len_result = sizeof(result) / sizeof(result[0]);
	if (isSortTrue(result) == false)
		return 0;
	else
		return 1;
}

//双机排序结果合并
void SortResultCombine(float* StrA, int lenA, float* StrB, int lenB, float* StrC) 
{
	int i, j, k;
	i = j = k = 0;
	while (i < lenA && j < lenB) {
		if (StrA[i] < StrB[j]) {
			StrC[k] = StrA[i];
			i++;
			k++;
		}
		else {
			StrC[k] = StrB[j];
			k++;
			j++;
		}
	}
	while (i < lenA) {
		StrC[k] = StrA[i];
		k++;
		i++;
	}
	while (j < lenB) {
		StrC[k] = StrB[j];
		k++;
		j++;
	}
}

void recvsort(int socket, float* remoteData) {
    int len;
    len = DATANUM;

    const int block_len = BUFFER_SIZE / sizeof(float);
    int recv_len = 0;
    int recv_size;
    while (recv_len < len) {
        
        recv_size = std::min(block_len, len - recv_len);
        ssize_t bytesRead = recv(socket, (void*)(remoteData + recv_len), recv_size * sizeof(float), 0);

        if(bytesRead <= 0){
            break;
        }
            
        recv_len += BUFFER_SIZE / sizeof(float);

    }


    // for (int i = 0; i < DATANUM / BUF_SIZE; i++) {
    //     retVal = recv(socket, buffer, BUF_SIZE, 0);

    //     startindex = i * BUF_SIZE / sizeof(float); // 调整为浮点数索引
    //     endindex = startindex + BUF_SIZE / sizeof(float);
    //     k = 0;

    //     for (int j = startindex; j < endindex; j++) {
    //         memcpy(temp, &buffer[k * sizeof(float)], sizeof(float));//从buffer中复制数据到temp
    //         remoteData[j] = atof(temp);//将以null结尾的字符串转换成浮点数
    //         k++;
    //     }

    //     //发送确认信息，否则出现一组数没传完就连着传另一组数了，但是会增加耗时
    //     const char ack[] = "ok";
    //     send(socket, ack, sizeof(ack), 0);
        
    //     if (std::floor((i)/double(DATANUM/BUF_SIZE)*100)!=now){
    //         now = std::floor((i)/double(DATANUM/BUF_SIZE)*100);
    //         std::cout << "已收到"<< now << "%数据" << std::endl;
    //     }
            
    // }
}


int main() {
    //TCP通信
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;//接受所有
    address.sin_port = htons(8080);

    //绑定套接字
    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 3);//监听
    std::cout << "Server is listening on port 8080" << std::endl;

    //数据初始化
    for (size_t i = 0; i < DATANUM; i++){
		rawFloatData[i] = float(i + DATANUM);
		rawFloatData_sortresult_local[i] = 0.0;
	}
    // 打乱数组
    std::random_device rd;  // 随机数种子
    std::mt19937 g(rd());   // Mersenne Twister 随机数生成器
    std::shuffle(rawFloatData, rawFloatData + DATANUM, g);//洗牌函数

    float buffer[100]={0};
    struct timespec start, end;//linux上获取时间的方式
    long seconds, nanoseconds;          
    double elapsed; 

    new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);//验证有没有成功连接
    if (new_socket < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    else{
        // int key = -1;//决定是否循环的钥匙，在双机间传递
        
        // std::cout << "输入数字1、2、3...表示测试组的序号\n如果想结束测试输入0\n";
        // std::cin >> key;
        double floatsum = 0.0;
        double floatmax = 0.0;
        double floatsum_speedup = 0.0;
        double floatmax_speedup = 0.0;
        int bag;
        
        recv(new_socket, (char*)&bag, sizeof(bag), 0);
        /*------------------------不加速部分-------------------------------*/
        //不加速求和
        clock_gettime(CLOCK_MONOTONIC, &start);
        floatsum = sum(rawFloatData, DATANUM);
        recv(new_socket, (char*)&bag, sizeof(bag), 0);
        buffer[0] = bag;
        clock_gettime(CLOCK_MONOTONIC, &end);
        // 计算持续时间
        seconds = end.tv_sec - start.tv_sec; // 秒
        nanoseconds = end.tv_nsec - start.tv_nsec; // 纳秒
        elapsed = seconds + nanoseconds*1e-9; // 总时间（秒） 
        //输出结果
        std::cout << "1PC_sum: " << floatsum << std::endl;
        std::cout << "2PC_sum: " << buffer[0] + floatsum << std::endl;
        std::cout << "2PC_sum_time: "<< elapsed << std::endl;
        double sum_common_time = elapsed;

        //不加速求最大值
        clock_gettime(CLOCK_MONOTONIC, &start);
        floatmax = max(rawFloatData, DATANUM);
        recv(new_socket, (char*)&bag, sizeof(bag), 0);
        buffer[1] = bag;
        clock_gettime(CLOCK_MONOTONIC, &end);
        std::cout << "1PC_max: " << floatmax << std::endl;
        if(buffer[1] > floatmax){
            floatmax = buffer[1];
        }
        // 计算持续时间
        seconds = end.tv_sec - start.tv_sec; // 秒
        nanoseconds = end.tv_nsec - start.tv_nsec; // 纳秒
        elapsed = seconds + nanoseconds*1e-9; // 总时间（秒）
        std::cout << "2PC_max: " << floatmax  << std::endl;  
        std::cout << "2PC_max_time: "<< elapsed << std::endl;
        double max_common_time = elapsed;


        //不加速排序
        clock_gettime(CLOCK_MONOTONIC, &start);
        int flag = sort(rawFloatData, DATANUM, rawFloatData_sortresult_local);
        if(flag == 0)
            std::cout << "sort error" << std::endl;
        else
            std::cout << "sort success" << std::endl;
        
        // 分块接收排序结果
        recvsort(new_socket, rawFloatData_sortresult_remote);
        std::cout << "接收排序结果成功" << std::endl;
        
        SortResultCombine(rawFloatData_sortresult_local, DATANUM, rawFloatData_sortresult_remote, DATANUM, rawFloatData_sortresult_combined);
        clock_gettime(CLOCK_MONOTONIC, &end);
        // 计算持续时间
        seconds = end.tv_sec - start.tv_sec; // 秒
        nanoseconds = end.tv_nsec - start.tv_nsec; // 纳秒
        elapsed = seconds + nanoseconds*1e-9; // 总时间（秒）
        //size_t len_combined=sizeof(rawFloatData_sortresult_combined)/sizeof(rawFloatData_sortresult_combined[0]);
        if(isSortTrue(rawFloatData_sortresult_combined) == false)
            std::cout << "sort_combined  error" << std::endl;
        else
            std::cout << "sort_combined success" << std::endl;
        std::cout << "2PC_sort_time: "<< elapsed << std::endl;
        double sort_common_time = elapsed;
        
        /*------------------------加速部分---------------------------------*/
        
        //send(new_socket, (char*)&key, sizeof(key), 0);
        //加速求和
        std::cout<<"开始加速求和了"<< std::endl;
        clock_gettime(CLOCK_MONOTONIC, &start);
        floatsum_speedup = sum_speedup(rawFloatData, DATANUM);
        recv(new_socket, (char*)&bag, sizeof(bag), 0);
        buffer[0] = bag;
        clock_gettime(CLOCK_MONOTONIC, &end);
        seconds = end.tv_sec - start.tv_sec; // 秒
        nanoseconds = end.tv_nsec - start.tv_nsec; // 纳秒
        elapsed = seconds + nanoseconds*1e-9; // 总时间（秒） 
        std::cout << "1PC_sum_speedup: " << floatsum_speedup << std::endl;
        std::cout << "2PC_sum_sp4.9eedup: " << buffer[0] + floatsum_speedup << std::endl;
        std::cout << "2PC_sum_speedup_time: "<< elapsed << std::endl;
        double sum_speedup_time = elapsed;
        
        
        std::cout<<"开始加速求最大值了"<< std::endl;
        clock_gettime(CLOCK_MONOTONIC, &start);
        floatmax_speedup = max_speedup(rawFloatData, DATANUM);
        recv(new_socket, (char*)&bag, sizeof(bag), 0);
        buffer[1] = bag;
        clock_gettime(CLOCK_MONOTONIC, &end);
        std::cout << "1PC_max_speedup: " << floatmax_speedup << std::endl;
        if(buffer[1] > floatmax_speedup){
            floatmax_speedup = buffer[1];
        }
        seconds = end.tv_sec - start.tv_sec; // 秒
        nanoseconds = end.tv_nsec - start.tv_nsec; // 纳秒
        elapsed = seconds + nanoseconds*1e-9; // 总时间（秒）
        std::cout << "2PC_max_speedup: " << floatmax_speedup  << std::endl;  
        std::cout << "2PC_max_speedup_time: "<< elapsed << std::endl;
        double max_speedup_time = elapsed;

        
        std::cout<<"开始加速排序了"<< std::endl;
        clock_gettime(CLOCK_MONOTONIC, &start);
        int flag_speedup = sort_speedup(rawFloatData, DATANUM, rawFloatData_sortresult_local);
        if(flag_speedup == 0)
            std::cout << "sort_speedup error" << std::endl;
        else
            std::cout << "sort_speedup success" << std::endl;
        // 分块接收排序结果

        //recvsort(new_socket, rawFloatData_sortresult_remote, DATANUM);
        std::thread recvThread(recvsort, new_socket, rawFloatData_sortresult_remote);
        recvThread.join();

        SortResultCombine_speedup(rawFloatData_sortresult_local, DATANUM, rawFloatData_sortresult_remote, DATANUM, rawFloatData_sortresult_combined);
        clock_gettime(CLOCK_MONOTONIC, &end);
        // 计算持续时间
        seconds = end.tv_sec - start.tv_sec; // 秒
        nanoseconds = end.tv_nsec - start.tv_nsec; // 纳秒
        elapsed = seconds + nanoseconds*1e-9; // 总时间（秒）
        flag_speedup = sort_speedup(rawFloatData, DATANUM, rawFloatData_sortresult_local);
        if(flag_speedup == false)
            std::cout << "sort_combined_speedup  error" << std::endl;
        else
            std::cout << "sort_combined_speedup success" << std::endl;
        std::cout << "2PC_sort_speedup_time: "<< elapsed << std::endl;
        double sort_speedup_time = elapsed;


        std::cout << "============================================================" <<std::endl;
        std::cout << "项目           |不加速结果(s)    |加速结果(s)       |加速比      " <<std::endl;
        std::cout << "------------------------------------------------------------" <<std::endl;
        std::cout << std::left << std::setw(18) << "求和" << std::left << std::setw(18) << sum_common_time << std::left << std::setw(18) << sum_speedup_time << std::left << std::setw(18) << sum_common_time/sum_speedup_time << std::endl;
        std::cout << std::left << std::setw(19) << "最大值" << std::left << std::setw(18) << max_common_time << std::left << std::setw(18) << max_speedup_time << std::left << std::setw(18) << max_common_time/max_speedup_time << std::endl;
        std::cout << std::left << std::setw(18) << "排序" << std::left << std::setw(18) << sort_common_time << std::left << std::setw(18) << sort_speedup_time << std::left << std::setw(18) << sort_common_time/sort_speedup_time << std::endl;
        std::cout << "============================================================" <<std::endl;
        

    }

    close(new_socket);
    close(server_fd);

    return 0;
}
