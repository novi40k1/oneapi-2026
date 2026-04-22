#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    constexpr size_t TILE = 16;  // размер блока
    const size_t N = size;

    sycl::queue q(device, {sycl::property::queue::in_order()});

    float* d_a = sycl::malloc_device<float>(N * N, q);
    float* d_b = sycl::malloc_device<float>(N * N, q);
    float* d_c = sycl::malloc_device<float>(N * N, q);

    q.memcpy(d_a, a.data(), N * N * sizeof(float));
    q.memcpy(d_b, b.data(), N * N * sizeof(float));
    q.memset(d_c, 0, N * N * sizeof(float));

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> tileA({TILE, TILE}, h);
        sycl::local_accessor<float, 2> tileB({TILE, TILE}, h);

        h.parallel_for(
            sycl::nd_range<2>({N, N}, {TILE, TILE}),
            [=](sycl::nd_item<2> item) {
                size_t gi = item.get_global_id(0);
                size_t gj = item.get_global_id(1);
                size_t li = item.get_local_id(0);
                size_t lj = item.get_local_id(1);

                float acc = 0.0f;

                // проходим по всем тайлам вдоль K
                for (size_t t = 0; t < N; t += TILE) {
                    tileA[li][lj] = d_a[gi * N + (t + lj)];
                    tileB[li][lj] = d_b[(t + li) * N + gj];

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < TILE; ++k) {
                        acc += tileA[li][k] * tileB[k][lj];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                d_c[gi * N + gj] = acc;
            });
    }).wait();

    std::vector<float> c(N * N);
    q.memcpy(c.data(), d_c, N * N * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return c;
}