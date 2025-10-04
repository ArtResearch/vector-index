cmake_minimum_required(VERSION 3.22)
project(metadata)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
find_package(Boost REQUIRED COMPONENTS program_options system url)
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

# Build OpenCV from third_party
set(BUILD_LIST "core,features,flann,imgproc,imgcodecs,calib" CACHE STRING "")
set(BUILD_opencv_apps OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_DOCS OFF CACHE BOOL "")
set(BUILD_PACKAGE OFF CACHE BOOL "")
set(CPU_DISPATCH "SSE4_1,SSE4_2,AVX,FP16,AVX2,AVX512_SKX" CACHE STRING "")
set(WITH_IPP OFF CACHE BOOL "")
add_subdirectory(third_party/opencv)

add_library(OpenCV::core ALIAS opencv_core)
add_library(OpenCV::features ALIAS opencv_features)
add_library(OpenCV::flann ALIAS opencv_flann)
add_library(OpenCV::imgproc ALIAS opencv_imgproc)
add_library(OpenCV::imgcodecs ALIAS opencv_imgcodecs)
add_library(OpenCV::calib ALIAS opencv_calib)

# Set march=native and AVX flags for the rest of the project and its dependencies (except OpenCV).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mavx512f -mavx512bw")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx2 -mavx512f -mavx512bw")


add_executable(CreateWorkData src/CreateWorkData.cpp)

target_include_directories(CreateWorkData PUBLIC 
    third_party/lazycsv/include
)

target_link_libraries(CreateWorkData PRIVATE Boost::program_options OpenMP::OpenMP_CXX)
target_compile_options(CreateWorkData PRIVATE -Wno-unknown-pragmas)

add_executable(createUSearchIndex src/createUSearchIndex.cpp)

option(USEARCH_USE_SIMSIMD "Use SimSIMD hardware-accelerated metrics" ON)
option(USEARCH_USE_OPENMP "Use OpenMP for a thread pool" ON)

FetchContent_Declare(
    usearch
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/usearch
)
FetchContent_MakeAvailable(usearch)


target_include_directories(createUSearchIndex PUBLIC
    third_party/libnpy/include
)

target_link_libraries(createUSearchIndex PRIVATE Boost::program_options usearch::usearch OpenMP::OpenMP_CXX)
target_compile_options(createUSearchIndex PRIVATE -Wno-unknown-pragmas)

add_executable(searchUSearchIndex src/searchUSearchIndex.cpp)

target_include_directories(searchUSearchIndex PUBLIC
    third_party/libnpy/include
    third_party/lazycsv/include
)

target_link_libraries(searchUSearchIndex PRIVATE Boost::program_options usearch::usearch OpenMP::OpenMP_CXX)
target_compile_options(searchUSearchIndex PRIVATE -Wno-unknown-pragmas)

add_executable(searchUSearchIndex_server src/searchUSearchIndex_server.cpp)

target_include_directories(searchUSearchIndex_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    third_party/lazycsv/include
)

target_link_libraries(searchUSearchIndex_server PRIVATE
    Boost::program_options
    Boost::system
    Boost::url
    usearch::usearch
    OpenMP::OpenMP_CXX
    OpenCV::core
    OpenCV::features
    OpenCV::imgcodecs
    OpenCV::calib
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
)
target_compile_options(searchUSearchIndex_server PRIVATE -Wno-unknown-pragmas)

option(MEASURE_PERFORMANCE "Enable performance measurements" ON)
if(MEASURE_PERFORMANCE)
  target_compile_definitions(searchUSearchIndex_server PRIVATE MEASURE_PERFORMANCE)
endif()

option(DEBUG_SIFT "Enable SIFT debug logging" ON)
if(DEBUG_SIFT)
  target_compile_definitions(searchUSearchIndex_server PRIVATE DEBUG_SIFT)
endif()

add_executable(mergeData src/mergeData.cpp)

target_include_directories(mergeData PUBLIC
    third_party/libnpy/include
    third_party/lazycsv/include
)

target_link_libraries(mergeData PRIVATE Boost::program_options)
target_compile_options(mergeData PRIVATE -Wno-unknown-pragmas)
