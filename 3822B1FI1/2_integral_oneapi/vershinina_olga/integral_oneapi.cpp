#include "integral_oneapi.h"
#include <vector>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue queue(device);
    float h = (end - start) / count;
    int local_count = count > 2000 ? 2000 : count;
    float local_h = (end - start) / local_count;
    std::vector<float> partial_sums(local_count, 0.0f);
    {
        sycl::buffer<float, 1> sum_buf(partial_sums.data(), sycl::range<1>(local_count));
        queue.submit([&](sycl::handler& cgh) {
            auto sums_acc = sum_buf.get_access<sycl::access::mode::read_write>(cgh);
            cgh.parallel_for(sycl::range<1>(local_count), [=](sycl::item<1> item) {
                size_t i = item[0];
                float x = start + (i + 0.5f) * local_h;
                float row_sum = 0.0f;
                for (int j = 0; j < local_count; ++j) {
                    float y = start + (j + 0.5f) * local_h;
                    row_sum += sycl::sin(x) * sycl::cos(y) * local_h * local_h;
                }
                sums_acc[i] = row_sum;
                });
            });
    }
    float result = 0.0f;
    for (float v : partial_sums) {
        result += v;
    }

    return result;
}