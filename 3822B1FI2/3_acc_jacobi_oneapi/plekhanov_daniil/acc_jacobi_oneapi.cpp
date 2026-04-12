#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    sycl::queue queue(device);
    
    int n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);
    
    sycl::buffer<float> a_buf(a.data(), a.size());
    sycl::buffer<float> b_buf(b.data(), b.size());
    sycl::buffer<float> x_buf(x.data(), x.size());
    sycl::buffer<float> x_new_buf(x_new.data(), x_new.size());
    
    bool converged = false;
    
    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                float a_ii = a_acc[i * n + i];
                
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_acc[i * n + j] * x_acc[j];
                    }
                }
                
                x_new_acc[i] = (b_acc[i] - sum) / a_ii;
            });
        });
        
        queue.wait();
        
        float diff_norm = 0.0f;
        {
            sycl::buffer<float> diff_buf(&diff_norm, 1);
            
            queue.submit([&](sycl::handler& cgh) {
                auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
                auto x_new_acc = x_new_buf.get_access<sycl::access::mode::read>(cgh);
                auto diff_acc = diff_buf.get_access<sycl::access::mode::write>(cgh);
                
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    int i = idx[0];
                    float diff = sycl::fabs(x_new_acc[i] - x_acc[i]);
                    
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device> atomic_diff(diff_acc[0]);
                    if (diff > atomic_diff.load()) {
                        atomic_diff.store(diff);
                    }
                });
            });
            
            queue.wait();
        }
        
        std::swap(x, x_new);
        
        {
            sycl::buffer<float> x_new_swap_buf(x_new.data(), x_new.size());
            x_buf = sycl::buffer<float>(x.data(), x.size());
            x_new_buf = sycl::buffer<float>(x_new.data(), x_new.size());
        }
        
        if (diff_norm < accuracy) {
            converged = true;
        }
    }
    
    {
        sycl::host_accessor x_host(x_buf, sycl::read_only);
        for (int i = 0; i < n; ++i) {
            x[i] = x_host[i];
        }
    }
    
    return x;
}