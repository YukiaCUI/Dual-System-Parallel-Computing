# Dual-System-Parallel-Computing
本大作业实现了linux系统上的Server端与windows系统上的Client端之间的通信，并行计算

# IP
两台电脑同时连接同一网络（推荐手机热点）
linux终端输入命令：ifconfig
获取IP地址，将此地址写在2151105cyj_Client的client.cpp
第279行server.sin_addr.s_addr = inet_addr("192.168.43.147");
将"192.168.43.147"内容替换。

# Server
2151106cyj_Server
在linux系统下使用cmake运行，具体方式如下：
在~/2151106cyj_Server打开终端

如果没有安装cmake：
sudo apt-get update
sudo apt-get install cmake
安装则跳过

mkdir build
cd build
cmake ..
make
./TCPServer
即可运行

# Client
2151105cyj_Client
在windows系统下运行，可以用Visual Studio直接打开
成功运行Server.cpp后，直接用Visual Studio运行client.cpp

