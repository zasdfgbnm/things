#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>

int main() {
    thrust::device_ptr<int> p;
    thrust::adjacent_difference(p, p + 5, p, [=] __device__ (int64_t a, int64_t b) -> bool { return true; });
}
