#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device)
{
    if (count <= 0)
        return 0.0f;

    sycl::queue q(device);
    float step = (end - start) / static_cast<float>(count);
    float area = step * step;

    int count4 = (count / 4) * 4;
    int rem = count - count4;

    float result = 0.0f;
    {
        sycl::buffer<float> result_buf(&result, 1);

        q.submit([&](sycl::handler &cgh)
                 {
            auto reduction = sycl::reduction(result_buf, cgh, std::plus<>());

            cgh.parallel_for(
                sycl::range<2>(count4 / 4, count),
                reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    int block_x = idx[0];
                    int j = idx[1];

                    float x_base = start + (static_cast<float>(block_x * 4) + 0.5f) * step;
                    float y_mid = start + (static_cast<float>(j) + 0.5f) * step;
                    float cos_y = sycl::cos(y_mid);

                    sycl::float4 xs = x_base + sycl::float4(0.0f, step, 2.0f*step, 3.0f*step);
                    sycl::float4 sin_vals = sycl::sin(xs);

                    float value = (sin_vals.x() + sin_vals.y() + sin_vals.z() + sin_vals.w()) * cos_y * area;
                    sum += value;
                }
            ); });
        q.wait();
    }

    if (rem > 0)
    {
        float tail_sum = 0.0f;
        for (int i = count4; i < count; ++i)
        {
            float x_mid = start + (static_cast<float>(i) + 0.5f) * step;
            for (int j = 0; j < count; ++j)
            {
                float y_mid = start + (static_cast<float>(j) + 0.5f) * step;
                tail_sum += sycl::sin(x_mid) * sycl::cos(y_mid) * area;
            }
        }
        result += tail_sum;
    }

    return result;
}