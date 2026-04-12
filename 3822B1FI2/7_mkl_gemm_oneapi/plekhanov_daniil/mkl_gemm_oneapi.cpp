#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& left,
        const std::vector<float>& right,
        size_t size,
        sycl::device device) {

    sycl::queue exec_queue(device);

    const size_t dim = size;

    const float scaleA = 1.0f;
    const float scaleB = 0.0f;

    std::vector<float> output(dim * dim);

    {
        sycl::buffer<float, 1> buf_left(left.data(), sycl::range<1>(left.size()));
        sycl::buffer<float, 1> buf_right(right.data(), sycl::range<1>(right.size()));
        sycl::buffer<float, 1> buf_out(output.data(), sycl::range<1>(output.size()));

        oneapi::mkl::blas::row_major::gemm(
            exec_queue,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            dim,
            dim,
            dim,
            scaleA,
            buf_left,
            dim,
            buf_right,
            dim,
            scaleB,
            buf_out,
            dim
        );
    }

    return output;
}