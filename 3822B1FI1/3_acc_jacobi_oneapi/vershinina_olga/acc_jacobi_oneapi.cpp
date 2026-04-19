#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device) {

    sycl::queue queue(device);
    size_t n = b.size();

    std::vector<float> x_curr(n, 0.0f);
    std::vector<float> x_next(n, 0.0f);

    {
        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(n * n));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> x_curr_buf(x_curr.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> x_next_buf(x_next.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> diff_buf(1);

        int curr = 0;
        int next = 1;

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            queue.submit([&](sycl::handler& cgh) {
                auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
                auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
                auto x_c_acc = (curr == 0 ? x_curr_buf : x_next_buf).get_access<sycl::access::mode::read>(cgh);
                auto x_n_acc = (curr == 0 ? x_next_buf : x_curr_buf).get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for(sycl::range<1>(n), [=](sycl::item<1> item) {
                    size_t i = item[0];
                    float sum = 0.0f;
                    for (size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_acc[i * n + j] * x_c_acc[j];
                        }
                    }
                    x_n_acc[i] = (b_acc[i] - sum) / a_acc[i * n + i];
                    });
                });

            queue.submit([&](sycl::handler& cgh) {
                auto x_c_acc = (curr == 0 ? x_curr_buf : x_next_buf).get_access<sycl::access::mode::read>(cgh);
                auto x_n_acc = (curr == 0 ? x_next_buf : x_curr_buf).get_access<sycl::access::mode::read>(cgh);
                auto diff_acc = diff_buf.get_access<sycl::access::mode::write>(cgh);

                cgh.single_task([=]() {
                    float max_diff = 0.0f;
                    for (size_t i = 0; i < n; ++i) {
                        float d = sycl::fabs(x_n_acc[i] - x_c_acc[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    diff_acc[0] = max_diff;
                    });
                });

            queue.wait_and_throw();

            sycl::host_accessor diff_host(diff_buf, sycl::read_only);
            if (diff_host[0] < accuracy) {
                break;
            }

            std::swap(curr, next);
        }

        sycl::host_accessor res_acc((curr == 0 ? x_curr_buf : x_next_buf), sycl::read_only);
        for (size_t i = 0; i < n; ++i) {
            x_curr[i] = res_acc[i];
        }
    }

    return x_curr;
}