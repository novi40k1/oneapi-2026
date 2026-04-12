#include "jacobi_kokkos.h"

#include <limits>
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    
    int n = b.size();

    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> a_dev("a_dev", a.size());
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> b_dev("b_dev", n);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> x_dev("x_dev", n);
    Kokkos::View<float*, Kokkos::SYCLDeviceUSMSpace> x_new_dev("x_new_dev", n);

    auto a_host = Kokkos::create_mirror_view(a_dev);
    auto b_host = Kokkos::create_mirror_view(b_dev);
    auto x_host = Kokkos::create_mirror_view(x_dev);

    for (size_t i = 0; i < a.size(); ++i) {
        a_host(i) = a[i];
    }

    for (int i = 0; i < n; ++i) {
        b_host(i) = b[i];
        x_host(i) = 0.0f;
    }

    Kokkos::deep_copy(a_dev, a_host);
    Kokkos::deep_copy(b_dev, b_host);
    Kokkos::deep_copy(x_dev, x_host);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::SYCL>(0, n),
            KOKKOS_LAMBDA(int i) {

                float sum = 0.0f;
                float a_ii = a_dev(i * n + i);

                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_dev(i * n + j) * x_dev(j);
                    }
                }

                x_new_dev(i) = (b_dev(i) - sum) / a_ii;
            }
        );

        float diff_norm = -std::numeric_limits<float>::infinity();

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::SYCL>(0, n),
            KOKKOS_LAMBDA(int i, float& max_diff) {
                float diff = fabs(x_new_dev(i) - x_dev(i));
                if (diff > max_diff) {
                    max_diff = diff;
                }
            },
            Kokkos::Max<float>(diff_norm)
        );

        Kokkos::swap(x_dev, x_new_dev);

        if (diff_norm < accuracy) {
            converged = true;
        }
    }

    Kokkos::deep_copy(x_host, x_dev);

    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = x_host(i);
    }

    return x;
}