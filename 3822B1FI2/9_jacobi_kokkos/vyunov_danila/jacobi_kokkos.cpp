#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float> a,
        const std::vector<float> b,
        float accuracy) {

    using ExecSpace = Kokkos::SYCL;
    using MemSpace  = Kokkos::SYCLDeviceUSMSpace;

    const int n = static_cast<int>(b.size());

    Kokkos::View<float*, MemSpace> a_dev("a_dev", n * n);
    Kokkos::View<float*, MemSpace> b_dev("b_dev", n);
    Kokkos::View<float*, MemSpace> x_dev("x_dev", n);
    Kokkos::View<float*, MemSpace> x_new_dev("x_new_dev", n);

    auto a_host = Kokkos::create_mirror_view(a_dev);
    auto b_host = Kokkos::create_mirror_view(b_dev);

    for (int i = 0; i < n * n; ++i) a_host(i) = a[i];
    for (int i = 0; i < n;     ++i) b_host(i) = b[i];

    Kokkos::deep_copy(a_dev, a_host);
    Kokkos::deep_copy(b_dev, b_host);
    Kokkos::deep_copy(x_dev, 0.0f);
    Kokkos::deep_copy(x_new_dev, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for(
            "JacobiUpdate",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(int i) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_dev(i * n + j) * x_dev(j);
                    }
                }
                x_new_dev(i) = (b_dev(i) - sum) / a_dev(i * n + i);
            }
        );

        float max_diff = 0.0f;
        Kokkos::parallel_reduce(
            "JacobiError",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(int i, float& local_max) {
                float diff = Kokkos::fabs(x_new_dev(i) - x_dev(i));
                if (diff > local_max) local_max = diff;
            },
            Kokkos::Max<float>(max_diff)
        );

        Kokkos::swap(x_dev, x_new_dev);

        if (max_diff < accuracy) break;
    }

    auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_dev);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) result[i] = x_host(i);

    return result;
}
