//
// Created by bianzheng on 2023/7/18.
//
// Pybind11 library
#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/struct/TypeDef.hpp"
#include "impl/BruteForceProgressive.hpp"


namespace VectorSetSearch::Method {

    PYBIND11_MODULE(BruteForceProgressive, m) {  // NOLINT

        py::class_<BruteForceProgressive>(
                m, "DocRetrieval",
                "The DocRetrieval module allows you to build, query, save, and load a "
                "semantic document search index.")
                .def(py::init<vector_set_list, uint32_t>(),
                     py::arg("query_l"),
                     py::arg("vec_dim"))
                .def("computeScore", &BruteForceProgressive::computeQueryItemScore,
                     py::arg("item_l"), py::arg("itemID_l"))
                .def("searchKNN", &BruteForceProgressive::searchKNN,
                     py::arg("topk"))
                .def("finish_compute", &BruteForceProgressive::finishCompute)
                .def_static("merge_result", &BruteForceProgressive::mergeResult,
                            py::arg("final_distance_l"), py::arg("final_id_l"),
                            py::arg("distance_l"), py::arg("id_l"),
                            "Merge the retrieval result by multiple searching");
    }

}  // namespace VectorSetSearch::python