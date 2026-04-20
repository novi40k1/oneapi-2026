#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device)
{
    size_t n = b.size();
    std::vector<float> result(n);

    sycl::queue queue(device);

    float* shared_a = sycl::malloc_shared<float>(n * n, queue);
    float* shared_b = sycl::malloc_shared<float>(n, queue);
    float* shared_x = sycl::malloc_shared<float>(n, queue);
    float* shared_x_new = sycl::malloc_shared<float>(n, queue);

    for (size_t i = 0; i < n * n; ++i)
    {
        shared_a[i] = a[i];
    }
    for (size_t i = 0; i < n; ++i)
    {
        shared_b[i] = b[i];
        shared_x[i] = 0.0f;
    }

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        queue.submit([&](sycl::handler& cgh)
            {
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                    {
                        size_t i = idx[0];
                        float sum = shared_b[i];

                        for (size_t j = 0; j < n; ++j)
                        {
                            if (i != j)
                            {
                                sum -= shared_a[i * n + j] * shared_x[j];
                            }
                        }

                        shared_x_new[i] = sum / shared_a[i * n + i];
                    });
            });
        queue.wait();

        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::fabs(shared_x_new[i] - shared_x[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            shared_x[i] = shared_x_new[i];
        }

        if (max_diff < accuracy)
        {
            break;
        }
    }

    for (size_t i = 0; i < n; ++i)
    {
        result[i] = shared_x[i];
    }

    sycl::free(shared_a, queue);
    sycl::free(shared_b, queue);
    sycl::free(shared_x, queue);
    sycl::free(shared_x_new, queue);

    return result;
}