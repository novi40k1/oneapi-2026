#include "block_gemm_oneapi.h"

constexpr int BLOCK_SIZE = 16;

std::vector<float> GemmBlockONEAPI(
    const std::vector<float> &a, const std::vector<float> &b,
    size_t size, sycl::device device)
{

    if (size == 0)
        return {};
    size_t blocks = size / BLOCK_SIZE;

    std::vector<float> c(size * size, 0.0f);

    sycl::queue q(device);

    {
        sycl::buffer<float, 2> buf_a(a.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_b(b.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_c(c.data(), sycl::range<2>(size, size));

        q.submit([&](sycl::handler &cgh)
                 {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(cgh);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(cgh);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(cgh);

            sycl::local_accessor<float, 2> local_a(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::local_accessor<float, 2> local_b(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(size, size),
                    sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
                ),
                [=](sycl::nd_item<2> item) {
                    int bx = item.get_group(0);
                    int by = item.get_group(1);
                    int tx = item.get_local_id(0);
                    int ty = item.get_local_id(1);

                    int row = by * BLOCK_SIZE + ty;
                    int col = bx * BLOCK_SIZE + tx;

                    float sum = 0.0f;

                    for (int k = 0; k < blocks; ++k) {
                        local_a[ty][tx] = acc_a[row][k * BLOCK_SIZE + tx];
                        local_b[ty][tx] = acc_b[k * BLOCK_SIZE + ty][col];
                        item.barrier(sycl::access::fence_space::local_space);

                        for (int i = 0; i < BLOCK_SIZE; ++i) {
                            sum += local_a[ty][i] * local_b[i][tx];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    acc_c[row][col] = sum;
                }
            ); });
        q.wait();
    }

    return c;
}