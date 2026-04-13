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

# We want our own binaries to use jemalloc as well
find_library(JEMALLOC_LIB jemalloc REQUIRED)
message(STATUS "Found jemalloc: ${JEMALLOC_LIB}")

option(USEARCH_USE_SIMSIMD "Use SimSIMD hardware-accelerated metrics" ON)
FetchContent_Declare(
    usearch
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/usearch
)
FetchContent_MakeAvailable(usearch)
target_compile_options(usearch INTERFACE -Wno-narrowing)

# --- Existing Tools ---
add_executable(CreateWorkData src/CreateWorkData.cpp)
target_include_directories(CreateWorkData PUBLIC third_party/lazycsv/include)
target_link_libraries(CreateWorkData PRIVATE Boost::program_options OpenMP::OpenMP_CXX ${JEMALLOC_LIB})
target_compile_options(CreateWorkData PRIVATE -Wno-unknown-pragmas)

add_executable(createUSearchIndex src/createUSearchIndex.cpp)
target_include_directories(createUSearchIndex PUBLIC third_party/libnpy/include)
target_link_libraries(createUSearchIndex PRIVATE Boost::program_options usearch::usearch OpenMP::OpenMP_CXX ${JEMALLOC_LIB})
target_compile_options(createUSearchIndex PRIVATE -Wno-unknown-pragmas)

add_executable(mergeData src/mergeData.cpp)
target_include_directories(mergeData PUBLIC third_party/libnpy/include third_party/lazycsv/include)
target_link_libraries(mergeData PRIVATE Boost::program_options ${JEMALLOC_LIB})
target_compile_options(mergeData PRIVATE -Wno-unknown-pragmas)

# --- New Modular Search Server ---

# Common library (header-only)
add_library(ignis_common INTERFACE)
target_include_directories(ignis_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/index)
target_link_libraries(ignis_common INTERFACE usearch::usearch ${JEMALLOC_LIB})

# SPARQL parsing and formatting
add_library(ignis_sparql src/index/sparql/parser.cpp src/index/sparql/response_formatter.cpp)
target_include_directories(ignis_sparql PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/index)
target_link_libraries(ignis_sparql PUBLIC ignis_common)

# Image downloading and SIFT reranking
add_library(ignis_image src/index/image/downloader.cpp src/index/image/sift_reranker.cpp)
target_include_directories(ignis_image PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/index)
target_link_libraries(ignis_image PUBLIC ignis_common)
target_link_libraries(ignis_image PUBLIC ${OpenCV_LIBS}) # Changed to PUBLIC
target_link_libraries(ignis_image PRIVATE CURL::libcurl OpenMP::OpenMP_CXX)
option(MEASURE_PERFORMANCE "Enable performance measurements" ON)
if(MEASURE_PERFORMANCE)
  target_compile_definitions(ignis_image PRIVATE MEASURE_PERFORMANCE)
endif()
option(DEBUG_SIFT "Enable SIFT debug logging" ON)
if(DEBUG_SIFT)
  target_compile_definitions(ignis_image PRIVATE DEBUG_SIFT)
endif()

# Embedding client
add_library(ignis_embedding src/index/embedding/client.cpp)
target_include_directories(ignis_embedding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/index)
target_link_libraries(ignis_embedding PUBLIC ignis_common)
target_link_libraries(ignis_embedding PRIVATE Boost::system)

# Core search logic and data handling
add_library(ignis_search src/index/search/metadata.cpp src/index/search/service.cpp)
target_include_directories(ignis_search PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/index
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lazycsv/include
)
target_link_libraries(ignis_search PUBLIC ignis_common ignis_embedding ignis_image ignis_sparql)
target_link_libraries(ignis_search PRIVATE Boost::system)
if(MEASURE_PERFORMANCE)
  target_compile_definitions(ignis_search PRIVATE MEASURE_PERFORMANCE)
endif()

# HTTP server and request handlers
add_library(ignis_http src/index/http/server.cpp src/index/http/handlers.cpp)
target_include_directories(ignis_http PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/index)
target_link_libraries(ignis_http PUBLIC ignis_search ignis_sparql)
target_link_libraries(ignis_http PRIVATE Boost::system Boost::url)
if(MEASURE_PERFORMANCE)
  target_compile_definitions(ignis_http PRIVATE MEASURE_PERFORMANCE)
endif()

# Main executable
add_executable(usearchIndex src/index/app/main.cpp)
target_include_directories(usearchIndex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/index
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lazycsv/include
)
target_link_libraries(usearchIndex PRIVATE
    ignis_http
    ignis_search
    Boost::program_options
)
