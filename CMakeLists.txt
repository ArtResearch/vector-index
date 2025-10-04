cmake_minimum_required(VERSION 3.22)
project(metadata)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mavx512f -mavx512bw")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx2 -mavx512f -mavx512bw")
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
find_package(Boost REQUIRED COMPONENTS program_options system url)
find_package(OpenMP REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core features2d flann imgproc imgcodecs calib3d)

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

# Suppress narrowing conversion warnings specifically for the usearch library
target_compile_options(usearch INTERFACE -Wno-narrowing)

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
    ${OpenCV_LIBS}
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
