cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_C_COMPILER "gcc")

project(VectorSetSearch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

#add_compile_options(
#        -Wall
#        -Wextra
#        # TODO(Josh): Fix eigen error and add back -werror
#        -Wno-unused-function
#        -Wno-unused-parameter
#        -pedantic
#        -Ofast
#        -g
#        -fno-omit-frame-pointer
#)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green "${Esc}[32m")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("${Green}Release mode${ColourReset}")
    set(USE_AVX512 ON)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("${Green}Clang${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("${Green}GNU${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("${Green}MSVC${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS})
    else ()
        message("${Green}Unknown compiler${ColourReset}")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(USE_AVX512 OFF)
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fsanitize=address -lasan ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -DUSE_AVX -mavx ${OpenMP_CXX_FLAGS}")
    #    add_definitions(-DNDEBUG -march=native)
    message("${Green}Debug mode${ColourReset}")
else ()
    set(USE_AVX512 OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native ${OpenMP_CXX_FLAGS}")
    message("${Green}Unknown mode${ColourReset}")
endif ()

if (USE_AVX512)
    add_definitions(-DUSE_AVX512)

    # Set compiler options
    if (NOT MSVC)
        #  set(CMAKE_C_FLAGS "-O3 -ffast-math -march=native -mavx512f  -funroll-loops -fopenmp -g ${CMAKE_C_FLAGS}")
        set(CMAKE_C_FLAGS "-O3 -ffast-math -march=native -mavx512f -mavx512vl -mavx512bw -funroll-loops -fopenmp -g ${CMAKE_C_FLAGS}")

    else ()
        include("${CMAKE_CURRENT_LIST_DIR}/msvc_simd_isa.cmake")
        if (SUPPORT_MSVC_AVX512)
            set(MSVC_SIMD_FLAGS "/arch:AVX512")
        elseif (SUPPORT_MSVC_AVX2)
            set(MSVC_SIMD_FLAGS "/arch:AVX2")
        elseif (SUPPORT_MSVC_AVX)
            set(MSVC_SIMD_FLAGS "/arch:AVX")
        else ()
            set(MSVC_SIMD_FLAGS "")
        endif ()
        set(CMAKE_C_FLAGS "/O2 /fp:fast ${MSVC_SIMD_FLAGS} ${CMAKE_C_FLAGS}")
    endif ()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -Wall -Wextra -Wno-missing-braces")

endif ()

#add_library(sse4_1 INTERFACE IMPORTED)
#if (MSVC)
#    target_compile_options(sse4_1 INTERFACE /arch:SSE4.1)
#else ()
#    target_compile_options(sse4_1 INTERFACE -msse4.1)
#endif ()

#find_package(Python3 COMPONENTS Interpreter Development)

# Header only dependencies
#include_directories(deps/pybind/include)
#include_directories(deps/eigen)
#include_directories(deps/cereal/include)

#set(Eigen3_DIR /home/username1/software/eigen3_install/share/eigen3/cmake)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#find_package(Threads REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenBLAS REQUIRED)
set(BUILD_BFLOAT16 ON)

#add_subdirectory(deps/pybind)

# push the nvcc compiler path before the information
#set(Torch_DIR "$ENV{HOME}/software/anaconda3/envs/sim_search/lib/python3.8/site-packages/torch/share/cmake/Torch")
#message(${Torch_DIR})
#find_package(Torch REQUIRED)

#set(spdlog_DIR /home/username1/software/spdlog_install/usr/local/lib/cmake/spdlog)
find_package(spdlog REQUIRED)

#set(Parlay_DIR /home/username1/software/parlaylib_install/share/parlay/cmake)
find_package(Parlay REQUIRED)

#set(MKL_DIR ${PROJECT_SOURCE_DIR}/cmake)
#set(MKLROOT /opt/intel/oneapi/mkl/2024.2)
#find_package(MKL CONFIG REQUIRED)
#find_package(MKL CONFIG REQUIRED PATHS /opt/intel/oneapi/mkl/2024.2)

# Source dependencies
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR /home/username1/software/pybind11_install/usr/local/share/cmake/pybind11)
find_package(pybind11)

set(ZLIB_LIBRARY /usr/lib/x86_64-linux-gnu/libz.so)

add_subdirectory(baseline/emvb/external/faiss)
include_directories(baseline/emvb/external/faiss)

add_subdirectory(baseline/emvb/external/cnpy)
include_directories(baseline/emvb/external/cnpy)

# So we can include using from the base directory
include_directories(src)

find_package(OpenMP REQUIRED)

set(USE_CUDA ON)

if (USE_CUDA)
    message("${Green}use CUDA${ColourReset}")

    set(USE_RETRIEVAL_GPU ON)

    #    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_definitions(-DUSE_GPU)

    if (USE_RETRIEVAL_GPU)
        add_definitions(-DUSE_RETRIEVAL_GPU)
    endif ()

    #    set(CUDA_PATH /usr/local/cuda-12.6)
    find_package(CUDAToolkit)
#        set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
#    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
            set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
    #    set(CMAKE_CUDA_COMPILER /home/username1/software/anaconda3/envs/sim_search/bin/nvcc)
    #        set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
    #            set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)

    # when finding nvcc cannot compile even a simple program, set CMAKE_CUDA_ARCHITECTURES to different values
    #        set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_ARCHITECTURES 52)
    #        set(CMAKE_CUDA_ARCHITECTURES 35;50;72;86)

    enable_language(CUDA)
    #    find_package(CUDA)

    #        set(Thrust_DIR /usr/include/thrust)
    #    set(Thrust_DIR /mnt/c/ProgramData/Anaconda3/lib/cmake/thrust)
    set(Thrust_DIR /home/username2/software/thrust/thrust/cmake)
    set(libcudacxx_DIR /home/username2/software/libcudacxx/lib/cmake/libcudacxx)
    set(CUB_DIR /home/username2/software/cub/cub/cmake)
    find_package(Thrust REQUIRED CONFIG)
    thrust_create_target(Thrust)


    set_source_files_properties(src/app/BruteForceProgressive.cpp PROPERTIES LANGUAGE CUDA)
    pybind11_add_module(BruteForceProgressive src/app/BruteForceProgressive.cpp)
    #    target_include_directories(BruteForceProgressive PRIVATE /home/username2/software/anaconda3/envs/sim_search/include/python3.8)
    #    target_include_directories(BruteForceProgressive PRIVATE /usr/include/python3.10/include)
    target_link_libraries(BruteForceProgressive PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen cublas spdlog::spdlog ${OpenBLAS_LIB})
    target_include_directories(BruteForceProgressive PRIVATE ${OpenBLAS_INCLUDE_DIR})
    set_target_properties(BruteForceProgressive PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET BruteForceProgressive PROPERTY CXX_STANDARD "17")
    set_property(TARGET BruteForceProgressive PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET BruteForceProgressive PROPERTY CXX_EXTENSIONS OFF)
    # when showing nvcc fatal   : Unsupported gpu architecture 'compute_35', change CUDA_ARCHITECTURES
    set_target_properties(BruteForceProgressive PROPERTIES CUDA_ARCHITECTURES "52")
    target_compile_options(BruteForceProgressive PRIVATE -Xcompiler=-fopenmp)
    target_compile_definitions(BruteForceProgressive PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

else () # not use cuda

    pybind11_add_module(BruteForceProgressive src/app/BruteForceProgressive.cpp)
    target_link_libraries(BruteForceProgressive PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB})
    target_include_directories(BruteForceProgressive PRIVATE ${OpenBLAS_INCLUDE_DIR})

endif ()


pybind11_add_module(BruteForce src/app/BruteForce.cpp)
target_link_libraries(BruteForce PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB})
target_include_directories(BruteForce PRIVATE ${OpenBLAS_INCLUDE_DIR})

pybind11_add_module(IGP src/app/IGP.cpp)
target_link_libraries(IGP PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB})
target_include_directories(IGP PRIVATE ${OpenBLAS_INCLUDE_DIR})

pybind11_add_module(MUVERA src/app/MUVERA.cpp)
target_link_libraries(MUVERA PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB} Parlay::parlay)
target_include_directories(MUVERA PRIVATE ${OpenBLAS_INCLUDE_DIR})

#add_subdirectory(baseline/emvb)
#include_directories(baseline/emvb/include)

