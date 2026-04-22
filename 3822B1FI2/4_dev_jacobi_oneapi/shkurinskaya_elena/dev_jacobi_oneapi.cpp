#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());
    sycl::queue q(device, {sycl::property::queue::in_order()});

    float* d_a    = sycl::malloc_device<float>(n * n, q);
    float* d_b    = sycl::malloc_device<float>(n, q);
    float* d_x    = sycl::malloc_device<float>(n, q);
    float* d_xn   = sycl::malloc_device<float>(n, q);
    float* d_diff = sycl::malloc_device<float>(1, q);

    q.memcpy(d_a, a.data(), n * n * sizeof(float));
    q.memcpy(d_b, b.data(), n * sizeof(float));
    q.memset(d_x, 0, n * sizeof(float));

    float diff = 0.0f;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (j != i) sum += d_a[i * n + j] * d_x[j];
            }
            d_xn[i] = (d_b[i] - sum) / d_a[i * n + i];
        });

        q.submit([&](sycl::handler& h) {
            auto diff_red = sycl::reduction(
                d_diff, sycl::maximum<float>(),
                sycl::property::reduction::initialize_to_identity{});
            h.parallel_for(sycl::range<1>(n), diff_red,
                [=](sycl::id<1> idx, auto& d) {
                    d.combine(sycl::fabs(d_xn[idx] - d_x[idx]));
                });
        });

        q.memcpy(d_x, d_xn, n * sizeof(float));
        q.memcpy(&diff, d_diff, sizeof(float)).wait();

        if (diff < accuracy) break;
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), d_x, n * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_x, q);
    sycl::free(d_xn, q);
    sycl::free(d_diff, q);

    return result;
}