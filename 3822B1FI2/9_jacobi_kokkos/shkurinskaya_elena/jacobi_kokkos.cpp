#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    const int n = static_cast<int>(b.size());

    Kokkos::View<float*> d_a("a", n * n);
    Kokkos::View<float*> d_b("b", n);
    Kokkos::View<float*> d_x("x", n);
    Kokkos::View<float*> d_xn("xn", n);

    // копируем host -> device через mirror views
    auto ha = Kokkos::create_mirror_view(d_a);
    auto hb = Kokkos::create_mirror_view(d_b);
    for (int i = 0; i < n * n; ++i) ha(i) = a[i];
    for (int i = 0; i < n; ++i)     hb(i) = b[i];
    Kokkos::deep_copy(d_a, ha);
    Kokkos::deep_copy(d_b, hb);
    Kokkos::deep_copy(d_x, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("jacobi_update", n,
            KOKKOS_LAMBDA(int i) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (j != i) sum += d_a(i * n + j) * d_x(j);
                }
                d_xn(i) = (d_b(i) - sum) / d_a(i * n + i);
            });

        float diff = 0.0f;
        Kokkos::parallel_reduce("jacobi_diff", n,
            KOKKOS_LAMBDA(int i, float& upd) {
                float v = Kokkos::fabs(d_xn(i) - d_x(i));
                upd = upd > v ? upd : v;
            },
            Kokkos::Max<float>(diff));

        Kokkos::deep_copy(d_x, d_xn);

        if (diff < accuracy) break;
    }

    // копируем обратно на хост
    auto hx = Kokkos::create_mirror_view(d_x);
    Kokkos::deep_copy(hx, d_x);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) result[i] = hx(i);
    return result;
}