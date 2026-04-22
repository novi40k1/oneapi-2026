#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    float step = (end - start) / count;
    float result = 0.0f;

    // 2D range, reduce по сумме
    Kokkos::parallel_reduce(
        "integral_kokkos",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(int i, int j, float& sum) {
            float x = start + (i + 0.5f) * step;
            float y = start + (j + 0.5f) * step;
            sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        result);

    return result * step * step;
}