#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);
    std::vector<float> diff_host(1, 0.0f);

    sycl::queue q(device);

    {
        sycl::buffer<float> a_buf(a.data(), a.size());
        sycl::buffer<float> b_buf(b.data(), b.size());
        sycl::buffer<float> x_buf(x.data(), n);
        sycl::buffer<float> xn_buf(x_new.data(), n);
        sycl::buffer<float> diff_buf(diff_host.data(), 1);

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            // 1) jacobi update: xn = f(x)
            q.submit([&](sycl::handler& h) {
                auto A = a_buf.get_access<sycl::access::mode::read>(h);
                auto B = b_buf.get_access<sycl::access::mode::read>(h);
                auto X = x_buf.get_access<sycl::access::mode::read>(h);
                auto Xn = xn_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    int i = idx[0];
                    float sum = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (j != i) sum += A[i * n + j] * X[j];
                    }
                    Xn[i] = (B[i] - sum) / A[i * n + i];
                });
            });

            // 2) diff = max|xn - x|
            q.submit([&](sycl::handler& h) {
                auto X = x_buf.get_access<sycl::access::mode::read>(h);
                auto Xn = xn_buf.get_access<sycl::access::mode::read>(h);
                auto diff_red = sycl::reduction(
                    diff_buf, h, sycl::maximum<float>(),
                    sycl::property::reduction::initialize_to_identity{});

                h.parallel_for(sycl::range<1>(n), diff_red,
                    [=](sycl::id<1> idx, auto& d) {
                        d.combine(sycl::fabs(Xn[idx] - X[idx]));
                    });
            });

            // 3) x = xn
            q.submit([&](sycl::handler& h) {
                auto X = x_buf.get_access<sycl::access::mode::write>(h);
                auto Xn = xn_buf.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    X[idx] = Xn[idx];
                });
            });

            // читаем diff с хоста (host_accessor синхронизирует с GPU)
            float diff;
            {
                sycl::host_accessor d(diff_buf, sycl::read_only);
                diff = d[0];
            }
            if (diff < accuracy) break;
        }
    }

    return x;
}