#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());
    sycl::queue q(device, {sycl::property::queue::in_order()});

    // shared память — видна и с хоста, и с девайса
    float* s_a    = sycl::malloc_shared<float>(n * n, q);
    float* s_b    = sycl::malloc_shared<float>(n, q);
    float* s_x    = sycl::malloc_shared<float>(n, q);
    float* s_xn   = sycl::malloc_shared<float>(n, q);
    float* s_diff = sycl::malloc_shared<float>(1, q);

    // просто присваиваем — никаких memcpy не надо
    for (int i = 0; i < n * n; ++i) s_a[i] = a[i];
    for (int i = 0; i < n; ++i)     s_b[i] = b[i];
    for (int i = 0; i < n; ++i)     s_x[i] = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (j != i) sum += s_a[i * n + j] * s_x[j];
            }
            s_xn[i] = (s_b[i] - sum) / s_a[i * n + i];
        });

        q.submit([&](sycl::handler& h) {
            auto diff_red = sycl::reduction(
                s_diff, sycl::maximum<float>(),
                sycl::property::reduction::initialize_to_identity{});
            h.parallel_for(sycl::range<1>(n), diff_red,
                [=](sycl::id<1> idx, auto& d) {
                    d.combine(sycl::fabs(s_xn[idx] - s_x[idx]));
                });
        });

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            s_x[idx] = s_xn[idx];
        });

        q.wait();  // чтобы *s_diff был актуален на хосте
        if (*s_diff < accuracy) break;
    }

    std::vector<float> result(s_x, s_x + n);

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_x, q);
    sycl::free(s_xn, q);
    sycl::free(s_diff, q);

    return result;
}