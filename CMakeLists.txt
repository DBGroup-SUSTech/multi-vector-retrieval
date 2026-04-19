cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_C_COMPILER "gcc")

project(VectorSetSearch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

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

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#find_package(Threads REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenBLAS REQUIRED)
set(BUILD_BFLOAT16 ON)

set(spdlog_DIR /home/bianzheng/software/spdlog_install/usr/local/lib/cmake/spdlog)
find_package(spdlog REQUIRED)

set(TBB_DIR /usr/lib/x86_64-linux-gnu/cmake/TBB)
find_package(TBB REQUIRED)

# Source dependencies
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR /home/bianzheng/software/pybind11_install/usr/local/share/cmake/pybind11)
find_package(pybind11)

set(ZLIB_LIBRARY /usr/lib/x86_64-linux-gnu/libz.so)

add_subdirectory(deps/cmd_line_parser)
include_directories(deps/cmd_line_parser/include)

add_subdirectory(deps/cnpy)
include_directories(deps/cnpy)

include_directories(src)

find_package(OpenMP REQUIRED)

set(USE_CUDA ON)

if (USE_CUDA)
    message("${Green}use CUDA${ColourReset}")

    #    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_definitions(-DUSE_GPU)

    set(USE_RETRIEVAL_GPU ON)
    if (USE_RETRIEVAL_GPU)
        add_definitions(-DUSE_RETRIEVAL_GPU)
    endif ()

    set(OLD_GPU OFF)
    if (OLD_GPU)
        add_definitions(-DOLD_GPU)
    endif ()

    find_package(CUDAToolkit)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)

    # when finding nvcc cannot compile even a simple program, set CMAKE_CUDA_ARCHITECTURES to different values
    #        set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_ARCHITECTURES 52)
    #        set(CMAKE_CUDA_ARCHITECTURES 35;50;72;86)

    enable_language(CUDA)
    #    find_package(CUDA)

    set(CONDA_ENV_PATH /home/bianzheng/software/anaconda3/envs/billion_MVR)

    set(Thrust_DIR /home/bianzheng/software/thrust/thrust/cmake)
    set(libcudacxx_DIR /home/bianzheng/software/libcudacxx/lib/cmake/libcudacxx)
    set(CUB_DIR /home/bianzheng/software/cub/cub/cmake)
    set(rmm_DIR "${CONDA_ENV_PATH}/lib/cmake/rmm")
    set(nvtx3_DIR "${CONDA_ENV_PATH}/lib/cmake/nvtx3")
    set(cuco_DIR "${CONDA_ENV_PATH}/lib/cmake/cuco")
    set(rapids_logger_DIR "${CONDA_ENV_PATH}/lib/cmake/rapids_logger")

    set(raft_DIR "${CONDA_ENV_PATH}/lib/cmake/raft")
    #    set(raft_INCLUDE_DIR "${CONDA_ENV_PATH}/lib/include")
    #    find_package(rmm REQUIRED HINTS "${CONDA_ENV_PATH}/lib/cmake/rmm")
    find_package(raft REQUIRED HINTS "${CONDA_ENV_PATH}/lib/cmake/raft")

    set(hnswlib_DIR "${CONDA_ENV_PATH}/lib/cmake/hnswlib")
    find_package(cuvs REQUIRED HINTS "${CONDA_ENV_PATH}/lib/cmake/cuvs")

    set_source_files_properties(src/app/build_index/GPGPUBuildIndexPackGPU.cpp PROPERTIES LANGUAGE CUDA)
    pybind11_add_module(GPGPUBuildIndexPackGPU src/app/build_index/GPGPUBuildIndexPackGPU.cpp)
    #    target_include_directories(GPGPUBuildIndexPackGPU PRIVATE /home/bianzheng/software/anaconda3/envs/sim_search/include/python3.8)
    #    target_include_directories(GPGPUBuildIndexPackGPU PRIVATE /usr/include/python3.10/include)
    target_link_libraries(GPGPUBuildIndexPackGPU PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen cublas spdlog::spdlog ${OpenBLAS_LIB})
    target_include_directories(GPGPUBuildIndexPackGPU PRIVATE ${OpenBLAS_INCLUDE_DIR})
    set_target_properties(GPGPUBuildIndexPackGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET GPGPUBuildIndexPackGPU PROPERTY CXX_STANDARD "17")
    set_property(TARGET GPGPUBuildIndexPackGPU PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET GPGPUBuildIndexPackGPU PROPERTY CXX_EXTENSIONS OFF)
    # when showing nvcc fatal   : Unsupported gpu architecture 'compute_35', change CUDA_ARCHITECTURES
    set_target_properties(GPGPUBuildIndexPackGPU PROPERTIES CUDA_ARCHITECTURES "52")
    target_compile_options(GPGPUBuildIndexPackGPU PRIVATE -Xcompiler=-fopenmp)
    target_compile_definitions(GPGPUBuildIndexPackGPU PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


    set_source_files_properties(src/app/BruteForceProgressive.cpp PROPERTIES LANGUAGE CUDA)
    pybind11_add_module(BruteForceProgressive src/app/BruteForceProgressive.cpp)
    #    target_include_directories(BruteForceProgressive PRIVATE /home/bianzheng/software/anaconda3/envs/sim_search/include/python3.8)
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


    set_source_files_properties(src/app/IGPGPUPP.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(IGPGPUPP src/app/IGPGPUPP.cpp)
    #    target_include_directories(IGPGPUPP PRIVATE /home/bianzheng/software/anaconda3/envs/sim_search/include/python3.8)
    #    target_include_directories(IGPGPUPP PRIVATE /usr/include/python3.10/include)
    target_link_libraries(IGPGPUPP PRIVATE cnpy OpenMP::OpenMP_CXX Eigen3::Eigen cublas spdlog::spdlog ${OpenBLAS_LIB} cublasLt TBB::tbb)
    target_include_directories(IGPGPUPP PRIVATE ${OpenBLAS_INCLUDE_DIR} "${CONDA_ENV_PATH}/include")
    set_target_properties(IGPGPUPP PROPERTIES CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            INTERFACE_POSITION_INDEPENDENT_CODE ON
            CXX_EXTENSIONS OFF
            CUDA_ARCHITECTURES 52)
    target_compile_options(IGPGPUPP PRIVATE -Xcompiler=-fopenmp --expt-extended-lambda -lnvToolsExt --expt-relaxed-constexpr
            -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
    target_compile_definitions(IGPGPUPP PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

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
