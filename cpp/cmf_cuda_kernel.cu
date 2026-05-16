#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void euler_integrate_kernel(
    const scalar_t* __restrict__ z0,
    const scalar_t* __restrict__ velocity,
    scalar_t* __restrict__ out,
    const int64_t batch,
    const int64_t steps,
    const int64_t dim,
    const double dt) {
  const int64_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t lanes = batch * dim;
  if (lane >= lanes) {
    return;
  }

  const int64_t b = lane / dim;
  const int64_t d = lane % dim;
  scalar_t acc = z0[b * dim + d];

  for (int64_t s = 0; s < steps; ++s) {
    const int64_t idx = (b * steps + s) * dim + d;
    acc = acc + static_cast<scalar_t>(dt) * velocity[idx];
    out[idx] = acc;
  }
}

torch::Tensor euler_integrate_cuda(torch::Tensor z0, torch::Tensor velocity, double dt) {
  TORCH_CHECK(z0.dim() == 2, "z0 must have shape [batch, dim]");
  TORCH_CHECK(velocity.dim() == 3, "velocity must have shape [batch, steps, dim]");
  TORCH_CHECK(z0.is_cuda(), "z0 must be CUDA");
  TORCH_CHECK(velocity.is_cuda(), "velocity must be CUDA");
  TORCH_CHECK(z0.scalar_type() == velocity.scalar_type(), "dtype mismatch");
  TORCH_CHECK(z0.size(0) == velocity.size(0), "batch size mismatch");
  TORCH_CHECK(z0.size(1) == velocity.size(2), "latent dimension mismatch");

  auto z0_c = z0.contiguous();
  auto velocity_c = velocity.contiguous();
  auto out = torch::empty_like(velocity_c);

  const int64_t batch = velocity_c.size(0);
  const int64_t steps = velocity_c.size(1);
  const int64_t dim = velocity_c.size(2);
  const int threads = 256;
  const int64_t lanes = batch * dim;
  const int blocks = static_cast<int>((lanes + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(velocity_c.scalar_type(), "euler_integrate_cuda", [&] {
    euler_integrate_kernel<scalar_t><<<blocks, threads>>>(
        z0_c.data_ptr<scalar_t>(),
        velocity_c.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        batch,
        steps,
        dim,
        dt);
  });

  return out;
}

