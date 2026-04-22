#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float> &a, const std::vector<float> &b,
    float accuracy, sycl::device device)
{

    size_t N = b.size();
    if (N == 0)
        return {};

    sycl::queue q(device);

    float *shared_a = sycl::malloc_shared<float>(a.size(), q);
    float *shared_b = sycl::malloc_shared<float>(N, q);
    float *shared_x_old = sycl::malloc_shared<float>(N, q);
    float *shared_x_new = sycl::malloc_shared<float>(N, q);

    if (!shared_a || !shared_b || !shared_x_old || !shared_x_new)
    {
        if (shared_a)
            sycl::free(shared_a, q);
        if (shared_b)
            sycl::free(shared_b, q);
        if (shared_x_old)
            sycl::free(shared_x_old, q);
        if (shared_x_new)
            sycl::free(shared_x_new, q);
        return {};
    }

    std::copy(a.begin(), a.end(), shared_a);
    std::copy(b.begin(), b.end(), shared_b);
    std::fill(shared_x_old, shared_x_old + N, 0.0f);
    std::fill(shared_x_new, shared_x_new + N, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        q.submit([&](sycl::handler &cgh)
                 { cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
                                    {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (j != i) {
                        sum += shared_a[i * N + j] * shared_x_old[j];
                    }
                }
                float diag = shared_a[i * N + i];
                shared_x_new[i] = (shared_b[i] - sum) / diag; }); });
        q.wait();

        float max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i)
        {
            float diff = std::fabs(shared_x_new[i] - shared_x_old[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        if (max_diff < accuracy)
            break;

        q.submit([&](sycl::handler &cgh)
                 { cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
                                    { shared_x_old[idx] = shared_x_new[idx]; }); });
        q.wait();
    }

    std::vector<float> result(N);
    std::copy(shared_x_new, shared_x_new + N, result.begin());

    sycl::free(shared_a, q);
    sycl::free(shared_b, q);
    sycl::free(shared_x_old, q);
    sycl::free(shared_x_new, q);

    return result;
}