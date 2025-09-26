//
// Created by Administrator on 2025/9/19.
//
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>

#include "include/struct/TypeDef.hpp"
#include "impl/IGP.hpp"
#include "include/alg/refine/ResidualScalarQuantizationPack.hpp"

namespace VectorSetSearch::Method {

PYBIND11_MODULE(IGP, m) {  // NOLINT
  m.def("compute_quantized_scalar", &ComputeQuantizedScalarPack,
        py::arg("item_vec_l"), py::arg("centroid_l"), py::arg("code_l"),
        py::arg("n_bit"));

  py::class_<CompressResidualCode>(m, "CompressResidualCode",
                                   "The DocRetrieval module allows you to build, query, save, and load a "
                                   "semantic document search index.")
      .def(py::init<const pyarray_float&, const pyarray_float&, const pyarray_float&, const uint32_t>(),
           py::arg("centroid_l"), py::arg("cutoff_l"), py::arg("weight_l"),
           py::arg("n_bit"))
      .def_readonly("n_val_per_vec", &CompressResidualCode::n_uint8_per_vec_)
      .def("compute_residual_code",
           &CompressResidualCode::compute_residual_code,
           py::arg("vec_l"), py::arg("code_l"));

  py::class_<IGP>(m, "DocRetrieval",
                                "The DocRetrieval module allows you to build, query, save, and load a "
                                "semantic document search index.")
          .def(py::init<const std::vector<uint32_t> &,
                       const uint32_t &, const uint32_t &,
                       const uint32_t &, const uint32_t &>(),
               py::arg("item_n_vec_l"),
               py::arg("n_item"), py::arg("vec_dim"),
               py::arg("n_centroid"), py::arg("n_bit"))
          .def("load_quantization_index", &IGP::loadQuantizationIndex,
               py::arg("centroid_l"), py::arg("vq_code_l"),
               py::arg("weight_l"), py::arg("residual_code_l"))
          .def("build_graph_index", &IGP::buildGraphIndex)
          .def("save_graph_index", &IGP::saveGraphIndex, py::arg("filename"))
          .def("load_graph_index", &IGP::loadGraphIndex, py::arg("filename"))
          .def("search", &IGP::search,
               py::arg("query_l"), py::arg("topk"),
               py::arg("nprobe"), py::arg("probe_topk"));

}

}  // namespace VectorSetSearch::python