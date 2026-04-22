#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    const size_t N = size;
    sycl::queue q(device);

    float* d_a = sycl::malloc_device<float>(N * N, q);
    float* d_b = sycl::malloc_device<float>(N * N, q);
    float* d_c = sycl::malloc_device<float>(N * N, q);

    q.memcpy(d_a, a.data(), N * N * sizeof(float));
    q.memcpy(d_b, b.data(), N * N * sizeof(float));
    q.wait();

    // наши матрицы хранятся по строкам => используем row_major API
    // C = 1.0 * A * B + 0.0 * C
    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        N, N, N,
        1.0f,
        d_a, N,
        d_b, N,
        0.0f,
        d_c, N).wait();

    std::vector<float> c(N * N);
    q.memcpy(c.data(), d_c, N * N * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return c;
}