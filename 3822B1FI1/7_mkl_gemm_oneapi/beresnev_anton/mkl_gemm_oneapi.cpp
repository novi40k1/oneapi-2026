#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float> &a,
    const std::vector<float> &b,
    size_t size,
    sycl::device device)
{

    if (size == 0)
        return {};

    sycl::queue q(device);
    std::vector<float> c(size * size, 0.0f);

    sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> buf_c(c.data(), sycl::range<1>(c.size()));

    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size, size, size,
        1.0f,
        buf_a, size,
        buf_b, size,
        0.0f,
        buf_c, size);

    q.wait();
    return c;
}