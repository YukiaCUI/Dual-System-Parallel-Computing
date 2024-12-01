cmake_minimum_required(VERSION 3.0)

project(TCPServerClient)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create executable for TCP server
add_executable(TCPServer Server.cpp)



find_package(Threads REQUIRED)
target_link_libraries(TCPServer PRIVATE Threads::Threads)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TCPServer PUBLIC OpenMP::OpenMP_CXX)
    
endif()


# 检查编译器并添加 AVX 支持
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
    set_target_properties(TCPServer PROPERTIES 
                          COMPILE_FLAGS "-mavx"
                          )
endif()
