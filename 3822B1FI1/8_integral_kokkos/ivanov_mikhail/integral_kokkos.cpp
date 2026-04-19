#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    if (start == end || count <= 0)
        return 0.0;

    if (start > end)
        std::swap(start, end);

    float result = 0.0f;
    float step = (end - start) / count;

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, float& local_sum) {
            float x = start + (i + 0.5f) * step;
            float y = start + (j + 0.5f) * step;
            local_sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        result);

    return result * step * step;
}