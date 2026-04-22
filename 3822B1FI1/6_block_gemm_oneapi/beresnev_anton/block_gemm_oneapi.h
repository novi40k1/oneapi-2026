#ifndef BLOCK_GEMM_ONEAPI_H
#define BLOCK_GEMM_ONEAPI_H

#include <vector>
#include <sycl/sycl.hpp>

std::vector<float> GemmBlockONEAPI(
    const std::vector<float> &a, const std::vector<float> &b,
    size_t size, sycl::device device);

#endif // BLOCK_GEMM_ONEAPI_H