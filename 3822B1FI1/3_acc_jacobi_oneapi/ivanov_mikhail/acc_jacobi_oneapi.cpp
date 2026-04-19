#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
  const std::vector<float>& a, const std::vector<float>& b,
  float accuracy, sycl::device device) {

  size_t n = b.size();
  std::vector<float> cur_x(n, 0.0f);
  std::vector<float> prev_x(n, 0.0f);

  sycl::queue q(device);

  sycl::buffer<float> buf_a(a.data(), a.size());
  sycl::buffer<float> buf_b(b.data(), b.size());
  sycl::buffer<float> buf_prev_x(prev_x.data(), prev_x.size());
  sycl::buffer<float> buf_cur_x(cur_x.data(), cur_x.size());

  float diff = 0.0f;
  
  for (size_t iteration = 0; iteration < ITERATIONS; iteration++) {
    sycl::buffer<float> buf_diff(&diff, sycl::range<1>(1));
    diff = 0.0f;

    {
      q.submit([&](sycl::handler& h) {
        auto in_a = buf_a.get_access<sycl::access::mode::read>(h);
        auto in_b = buf_b.get_access<sycl::access::mode::read>(h);
        auto in_prev_x = buf_prev_x.get_access<sycl::access::mode::read>(h);
        auto out_cur_x = buf_cur_x.get_access<sycl::access::mode::write>(h);

        auto reduction = sycl::reduction(buf_diff, h, sycl::plus<float>());

        h.parallel_for(sycl::range<1>(n), reduction, [=](sycl::id<1> idx, auto& sum_diff) {
          size_t i = idx[0];

          float res = 0.0f;
          for (size_t j = 0; j < n; j++) {
            if (i != j) {
              res += in_a[i * n + j] * in_prev_x[j];
            }
          }

          float new_x = (in_b[i] - res) / in_a[i * n + i];
          out_cur_x[i] = new_x;
          sum_diff += (new_x - in_prev_x[i]) * (new_x - in_prev_x[i]);
          });
        });
      q.wait();
    }

    if (diff < accuracy * accuracy)
      break;

    std::swap(buf_cur_x, buf_prev_x);
  }

  std::vector<float> result(n);
  {
    auto in_cur_x = buf_cur_x.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < n; i++) {
      result[i] = in_cur_x[i];
    }
  }

  return result;
}