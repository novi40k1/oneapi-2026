#include "integral_oneapi.h"

#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  const float delta = (end - start) / count;
  const float delta_product = delta * delta;

  double result = 0.0;
  sycl::queue queue(device);
  sycl::buffer<double> result_buf(&result, 1);

  queue.submit([&](sycl::handler &cgh) {
        auto reduction = sycl::reduction(result_buf, cgh, sycl::plus<double>());

        cgh.parallel_for(
            sycl::range<2>(count, count), reduction,
            [=](sycl::id<2> idx, auto &sum) {
              const float x =
                  start + delta * (static_cast<float>(idx[0]) + 0.5f);
              const float y =
                  start + delta * (static_cast<float>(idx[1]) + 0.5f);

              const float value = sycl::sin(x) * sycl::cos(y);

              sum += static_cast<double>(value);
            });
      }).wait();

  return static_cast<float>(result * delta_product);
}