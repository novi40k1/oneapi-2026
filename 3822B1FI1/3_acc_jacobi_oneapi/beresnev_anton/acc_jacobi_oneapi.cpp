#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float> &a, const std::vector<float> &b,
    float accuracy, sycl::device device)
{

    size_t N = b.size();
    if (N == 0)
        return {};

    std::vector<float> x_old_host(N, 0.0f);
    std::vector<float> x_new_host(N, 0.0f);

    sycl::queue q(device);

    sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> buf_x_old(x_old_host.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> buf_x_new(x_new_host.data(), sycl::range<1>(N));

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        q.submit([&](sycl::handler &cgh)
                 {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(cgh);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(cgh);
            auto acc_x_old = buf_x_old.get_access<sycl::access::mode::read>(cgh);
            auto acc_x_new = buf_x_new.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < N; ++j) {
                    if (j != i) {
                        sum += acc_a[i * N + j] * acc_x_old[j];
                    }
                }
                float diag = acc_a[i * N + i];
                acc_x_new[i] = (acc_b[i] - sum) / diag;
            }); });
        q.wait();

        {
            sycl::host_accessor host_x_old(buf_x_old, sycl::read_only);
            sycl::host_accessor host_x_new(buf_x_new, sycl::read_only);
            float max_diff = 0.0f;
            for (size_t i = 0; i < N; ++i)
            {
                float diff = std::fabs(host_x_new[i] - host_x_old[i]);
                if (diff > max_diff)
                    max_diff = diff;
            }
            if (max_diff < accuracy)
                break;
        }

        q.submit([&](sycl::handler &cgh)
                 {
            auto acc_x_old = buf_x_old.get_access<sycl::access::mode::write>(cgh);
            auto acc_x_new = buf_x_new.get_access<sycl::access::mode::read>(cgh);
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                acc_x_old[idx] = acc_x_new[idx];
            }); });
        q.wait();
    }

    sycl::host_accessor final_x(buf_x_new, sycl::read_only);
    return std::vector<float>(final_x.get_pointer(), final_x.get_pointer() + N);
}