//
// Created by username2 on 2023/2/16.
//
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/struct/TypeDef.hpp"
#include "impl/BruteForce.hpp"

namespace VectorSetSearch::Method {

    PYBIND11_MODULE(BruteForce, m) {  // NOLINT

        py::class_<BruteForce>(
                m, "DocRetrieval",
                "The DocRetrieval module allows you to build, query, save, and load a "
                "semantic document search index.")
                .def(py::init<uint32_t, uint32_t>(),
                     py::arg("n_item"),
                     py::arg("vec_dim"))
                .def("add_item", &BruteForce::addItem, py::arg("item"),
                     py::arg("itemID"))
                .def("add_item_batch", &BruteForce::addItemBatch,
                     py::arg("item_l"), py::arg("itemID_l"))
                .def("set_n_thread", &BruteForce::setNumThread, py::arg("n_thread"))
                .def("build_index", &BruteForce::buildIndex)
                .def("query", &BruteForce::searchKNN, py::arg("query_l"),
                     py::arg("topk"));
    }

}  // namespace VectorSetSearch::python