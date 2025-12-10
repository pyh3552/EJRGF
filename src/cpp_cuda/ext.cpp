#include <torch/extension.h>
#include "EJRGF.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("EJRGF", &EJRGF_L2G_CUDA);
}