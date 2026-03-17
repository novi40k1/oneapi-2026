#include "integral_kokkos.h"

#include <cmath>

float IntegralKokkos(float start, float end, int count) {
  const float d = (end - start) / count;
  float res = 0.0f;

  Kokkos::parallel_reduce(
    "Integral",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { count, count }),
    KOKKOS_LAMBDA(int i, int j, float& sum) {
      float x = start + d * (i + 0.5f);
      float y = start + d * (j + 0.5f);
      sum += sinf(x) * cosf(y);
    },
    res
  );

  return res * d * d;
}