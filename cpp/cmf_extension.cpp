#include <torch/extension.h>

torch::Tensor euler_integrate_cuda(torch::Tensor z0, torch::Tensor velocity, double dt);

torch::Tensor euler_integrate_cpu(torch::Tensor z0, torch::Tensor velocity, double dt) {
  TORCH_CHECK(z0.dim() == 2, "z0 must have shape [batch, dim]");
  TORCH_CHECK(velocity.dim() == 3, "velocity must have shape [batch, steps, dim]");
  TORCH_CHECK(z0.size(0) == velocity.size(0), "batch size mismatch");
  TORCH_CHECK(z0.size(1) == velocity.size(2), "latent dimension mismatch");

  return z0.unsqueeze(1) + torch::cumsum(velocity * dt, 1);
}

torch::Tensor euler_integrate(torch::Tensor z0, torch::Tensor velocity, double dt) {
  if (z0.is_cuda()) {
#ifdef WITH_CUDA
    return euler_integrate_cuda(z0, velocity, dt);
#else
    return euler_integrate_cpu(z0, velocity, dt);
#endif
  }
  return euler_integrate_cpu(z0, velocity, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("euler_integrate", &euler_integrate, "Euler integrate precomputed CMF velocities");
}

