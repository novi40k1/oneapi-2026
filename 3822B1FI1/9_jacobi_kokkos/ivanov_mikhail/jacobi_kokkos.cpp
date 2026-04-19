#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    const size_t size = b.size();

    using ExecSpace = Kokkos::SYCL;
    using MemSpace = Kokkos::SYCLDeviceUSMSpace;

    Kokkos::View<float*, MemSpace> dev_a("a", a.size());
    Kokkos::View<float*, MemSpace> dev_b("b", size);
    Kokkos::View<float*, MemSpace> dev_prev("prev_x", size);
    Kokkos::View<float*, MemSpace> dev_cur("cur_x", size);

    auto host_a = Kokkos::create_mirror_view(dev_a);
    auto host_b = Kokkos::create_mirror_view(dev_b);
    auto host_prev = Kokkos::create_mirror_view(dev_prev);

    for (size_t i = 0; i < a.size(); i++) {
        host_a(i) = a[i];
    }

    for (size_t i = 0; i < size; i++) {
        host_b(i)    = b[i];
        host_prev(i) = 0.0f;
    }

    Kokkos::deep_copy(dev_a, host_a);
    Kokkos::deep_copy(dev_b, host_b);
    Kokkos::deep_copy(dev_prev, host_prev);

    for (int iter = 0; iter < ITERATIONS; iter++) {

        float diff = 0.0f;

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(size)),
            KOKKOS_LAMBDA(const int i, float& local_diff) {

                float res = 0.0f;
                size_t row = static_cast<size_t>(i) * size;

                for (size_t j = 0; j < size; ++j) {
                    if (j != static_cast<size_t>(i)) {
                        res += dev_a(row + j) * dev_prev(j);
                    }
                }

                float new_x = (dev_b(i) - res) / dev_a(row + static_cast<size_t>(i));
                dev_cur(i) = new_x;
                local_diff += (new_x - dev_prev(i)) * (new_x - dev_prev(i));
            },
            diff
        );

        std::swap(dev_prev, dev_cur);

        if (diff < accuracy * accuracy) {
            break;
        }
    }

    Kokkos::fence();

    auto host_result = Kokkos::create_mirror_view(dev_prev);
    Kokkos::deep_copy(host_result, dev_prev);

    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = host_result(i);
    }

    return result;
}