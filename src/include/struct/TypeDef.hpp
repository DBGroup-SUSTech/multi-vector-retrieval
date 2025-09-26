//
// Created by bianzheng on 2023/2/10.
//

#ifndef VECTORSETSEARCH_TYPEDEF_HPP
#define VECTORSETSEARCH_TYPEDEF_HPP

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using pyarray_float = py::array_t<float, py::array::c_style | py::array::forcecast>;
using pyarray_double = py::array_t<double, py::array::c_style | py::array::forcecast>;
using pyarray_uint32 = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using pyarray_uint16 = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>;
using pyarray_int = py::array_t<int, py::array::c_style | py::array::forcecast>;
using pyarray_uint8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

using vector_set = py::array_t<float, py::array::c_style | py::array::forcecast>;
using vector_set_list = py::array_t<float, py::array::c_style | py::array::forcecast>;
using vecsID_o = size_t; // means the offset when aligning a vector

#endif //VECTORSETSEARCH_TYPEDEF_HPP
