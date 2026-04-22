#include "integral_kokkos.h"
#include <Kokkos_Core.hpp>
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) return 0.0f;

    float dx = (end - start) / static_cast<float>(count);
    float dy = (end - start) / static_cast<float>(count);
    float area = dx * dy;

    float result = 0.0f;

    Kokkos::parallel_reduce(
        "IntegralRiemann",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(int i, int j, float& sum) {
            float x_mid = start + (static_cast<float>(i) + 0.5f) * dx;
            float y_mid = start + (static_cast<float>(j) + 0.5f) * dy;
            sum += std::sin(x_mid) * std::cos(y_mid) * area;
        },
        result
    );

    return result;
}