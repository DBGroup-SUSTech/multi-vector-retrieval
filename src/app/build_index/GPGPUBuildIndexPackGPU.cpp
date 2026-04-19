//
// Created by Administrator on 7/2/2025.
//
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>

#include "include/struct/TypeDef.hpp"
#include "include/alg/refine/ResidualScalarQuantizationPack.hpp"
#include "include/alg/refine/ResidualScalarQuantizationPackGPU.hpp"
#include "include/struct/IVFIndex.hpp"

namespace VectorSetSearch::Method {

PYBIND11_MODULE(GPGPUBuildIndexPackGPU, m) {
  // NOLINT
  m.def("compute_quantized_scalar", &ComputeQuantizedScalarPack,
        py::arg("item_vec_l"), py::arg("centroid_l"), py::arg("code_l"),
        py::arg("n_bit"));

  py::class_<CompressResidualCodeGPU>(m, "CompressResidualCode",
                                      "The DocRetrieval module allows you to build, query, save, and load a "
                                      "semantic document search index.")
      .def(py::init<const pyarray_float&, const pyarray_float&,
                    const pyarray_float&,
                    const uint32_t>(),
           py::arg("centroid_l"), py::arg("cutoff_l"), py::arg("weight_l"),
           py::arg("n_bit"))
      .def_readonly("n_val_per_vec", &CompressResidualCodeGPU::n_packed_val_per_vec_)
      .def("compute_residual_code",
           &CompressResidualCodeGPU::compute_residual_code_gpu,
           py::arg("vec_l"), py::arg("code_l"));

  py::class_<IVFIndex>(
          m, "IVFIndex",
          "The DocRetrieval module allows you to build, query, save, and load a "
          "semantic document search index.")
      .def(py::init<std::vector<uint32_t>, std::vector<uint32_t>,
                    size_t, uint32_t, uint32_t>(),
           py::arg("vq_code_l"), py::arg("item_n_vec_l"),
           py::arg("n_vec"), py::arg("n_item"), py::arg("n_centroid"))
      .def("build", &IVFIndex::build)
      .def("saveIVFIndex", &IVFIndex::saveIVFIndex, py::arg("location"))
      .def("saveGroupIndex", &IVFIndex::saveGroupIndex, py::arg("location"));

}

} // namespace VectorSetSearch::python