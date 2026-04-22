#include "jacobi_kokkos.h"

#include <algorithm>
#include <cmath>
#include <vector>

std::vector<float> JacobiKokkos(
    const std::vector<float> &a,
    const std::vector<float> &b,
    float accuracy)
{

    const size_t n = b.size();
    if (n == 0)
        return {};
    if (a.size() != n * n)
        return {};

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExecSpace::memory_space;

    Kokkos::View<const float **, MemSpace> A("A", n, n);
    Kokkos::View<const float *, MemSpace> B("B", n);
    Kokkos::View<float *, MemSpace> x_curr("x_curr", n);
    Kokkos::View<float *, MemSpace> x_next("x_next", n);

    auto A_host = Kokkos::create_mirror_view(A);
    auto B_host = Kokkos::create_mirror_view(B);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            A_host(i, j) = a[i * n + j];
        }
        B_host(i) = b[i];
    }
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(B, B_host);

    Kokkos::deep_copy(x_curr, 0.0f);
    Kokkos::deep_copy(x_next, 0.0f);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        float max_diff = 0.0f;
        Kokkos::parallel_reduce(
            "JacobiStep",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(int i, float &local_max) {
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j)
                {
                    if (j != static_cast<size_t>(i))
                    {
                        sum += A(i, j) * x_curr(j);
                    }
                }
                float new_val = (B(i) - sum) / A(i, i);
                x_next(i) = new_val;
                float diff = Kokkos::fabs(new_val - x_curr(i));
                if (diff > local_max)
                    local_max = diff;
            },
            Kokkos::Max<float>(max_diff));

        if (max_diff < accuracy)
        {
            converged = true;
            break;
        }

        Kokkos::View<float *, MemSpace> tmp = x_curr;
        x_curr = x_next;
        x_next = tmp;
    }

    Kokkos::View<float *, Kokkos::HostSpace> result_host("result_host", n);
    Kokkos::deep_copy(result_host, converged ? x_curr : x_next);

    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i)
        result[i] = result_host(i);
    return result;
}