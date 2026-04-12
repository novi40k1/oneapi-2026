#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    sycl::queue queue(device);

    int n = b.size();

    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    float* a_dev = sycl::malloc_device<float>(a.size(), queue);
    float* b_dev = sycl::malloc_device<float>(b.size(), queue);
    float* x_dev = sycl::malloc_device<float>(n, queue);
    float* x_new_dev = sycl::malloc_device<float>(n, queue);

    queue.memcpy(a_dev, a.data(), a.size() * sizeof(float)).wait();
    queue.memcpy(b_dev, b.data(), b.size() * sizeof(float)).wait();
    queue.memset(x_dev, 0, n * sizeof(float)).wait();
    queue.memset(x_new_dev, 0, n * sizeof(float)).wait();

    float diff_norm = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        queue.parallel_for(
            sycl::range<1>(n),
            [=](sycl::id<1> idx) {

                int i = idx[0];
                float sum = 0.0f;
                float a_ii = a_dev[i * n + i];

                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_dev[i * n + j] * x_dev[j];
                    }
                }

                x_new_dev[i] = (b_dev[i] - sum) / a_ii;
            }
        ).wait();

        queue.memset(&diff_norm, 0, sizeof(float)).wait();

        queue.parallel_for(
            sycl::range<1>(n),
            [=](sycl::id<1> idx) {

                int i = idx[0];
                float diff = sycl::fabs(x_new_dev[i] - x_dev[i]);

                sycl::atomic_ref<float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device> atomic_max(diff_norm);

                float old = atomic_max.load();
                while (diff > old && !atomic_max.compare_exchange_strong(old, diff)) {
                }
            }
        ).wait();

        std::swap(x_dev, x_new_dev);

        if (diff_norm < accuracy) {
            break;
        }
    }

    queue.memcpy(x.data(), x_dev, n * sizeof(float)).wait();

    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(x_dev, queue);
    sycl::free(x_new_dev, queue);

    return x;
}