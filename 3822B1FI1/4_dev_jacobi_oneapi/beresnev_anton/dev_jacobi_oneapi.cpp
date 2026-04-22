#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevOneAPI(
    const std::vector<float> &a, const std::vector<float> &b,
    float accuracy, sycl::device device)
{

    size_t N = b.size();
    if (N == 0)
        return {};

    sycl::queue q(device);

    float *d_a = sycl::malloc_device<float>(a.size(), q);
    float *d_b = sycl::malloc_device<float>(N, q);
    float *d_x_old = sycl::malloc_device<float>(N, q);
    float *d_x_new = sycl::malloc_device<float>(N, q);

    if (!d_a || !d_b || !d_x_old || !d_x_new)
    {
        if (d_a)
            sycl::free(d_a, q);
        if (d_b)
            sycl::free(d_b, q);
        if (d_x_old)
            sycl::free(d_x_old, q);
        if (d_x_new)
            sycl::free(d_x_new, q);
        return {};
    }

    q.memcpy(d_a, a.data(), a.size() * sizeof(float)).wait();
    q.memcpy(d_b, b.data(), N * sizeof(float)).wait();
    q.fill(d_x_old, 0.0f, N).wait();
    q.fill(d_x_new, 0.0f, N).wait();

    std::vector<float> host_x_new(N, 0.0f);
    std::vector<float> host_x_old(N, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        q.submit([&](sycl::handler &cgh)
                 { cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
                                    {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (j != i) {
                        sum += d_a[i * N + j] * d_x_old[j];
                    }
                }
                float diag = d_a[i * N + i];
                d_x_new[i] = (d_b[i] - sum) / diag; }); });
        q.wait();

        q.memcpy(host_x_new.data(), d_x_new, N * sizeof(float)).wait();
        q.memcpy(host_x_old.data(), d_x_old, N * sizeof(float)).wait();

        float max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i)
        {
            float diff = std::fabs(host_x_new[i] - host_x_old[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        if (max_diff < accuracy)
            break;

        q.memcpy(d_x_old, d_x_new, N * sizeof(float)).wait();
    }

    std::vector<float> result(N);
    q.memcpy(result.data(), d_x_new, N * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_x_old, q);
    sycl::free(d_x_new, q);

    return result;
}