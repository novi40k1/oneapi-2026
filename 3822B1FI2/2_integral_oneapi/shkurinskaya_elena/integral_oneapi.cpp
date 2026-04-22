#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue q(device);

    float step = (end - start) / count;
    float result = 0.0f;

    // считаем сумму sin(x)*cos(y) в центрах ячеек через reduction
    {
        sycl::buffer<float> sum_buf(&result, 1);

        q.submit([&](sycl::handler& h) {
            auto sum_reduction =
                sycl::reduction(sum_buf, h, sycl::plus<float>());

            h.parallel_for(sycl::range<2>(count, count), sum_reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    int i = idx[0];
                    int j = idx[1];
                    // центр ячейки (middle Riemann sum)
                    float x = start + (i + 0.5f) * step;
                    float y = start + (j + 0.5f) * step;
                    sum += sycl::sin(x) * sycl::cos(y);
                });
        }).wait();
    }

    // домножаем на площадь одной ячейки один раз в конце
    return result * step * step;
}