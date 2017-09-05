/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define EIGEN_USE_GPU
#include "paddle/operators/transpose_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void transpose_kernel(int nthreads, T* in_data, T* out_data,
                                 int ndims, std::vector<int> axis,
                                 std::vector<int> in_offset,
                                 std::vector<int> out_offset) {
  int to_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    int from_index = 0;
    int temp = to_index;
    for (size_t i = 0; i < ndims; i++) {
      from_index += (temp / out_offset[i]) * in_offset[axis[i]];
      temp = temp % out_offset[i];
    }
    out_data[to_index] = in_data[from_index];
  }
}

template <typename T>
class TransposeCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use GPUPlace.");
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in_data = in->template data<T>();
    auto* out_data = out->template mutable_data<T>(context.GetPlace());
    auto axis = context.op_.GetAttr<std::vector<int>>("axis");
    auto in_dim = in->dims();
    auto out_dim = out->dims();
    auto data_size = product(in_dim);
    size_t ndims = in_dim.size();

    std::vector<int> in_offset(ndims, 1);
    std::vector<int> out_offset(ndims, 1);

    for (int i = ndims - 2; i >= 0; i--) {
      in_offset[i] = in_offset[i + 1] * in_dim[i + 1];
      out_offset[i] = out_offset[i + 1] * out_dim[i + 1];
    }
    int block = 512;
    int grid = (data_size + block - 1) / block;
    transpose_kernel << grid, block >> (data_size, in_data, out_data, ndims,
                                        axis, in_offset, out_offset);
  }
};

template <T>
class TransposeGradCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use GPUPlace.");
    auto* in = context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* in_data = in->template data<T>();
    auto* out_data = out->template mutable_data<T>(context.GetPlace());
    auto axis_temp = context.op_.GetAttr<std::vector<int>>("axis");
    auto in_dim = in->dims();
    auto out_dim = out->dims();
    auto data_size = product(in_dim);
    size_t ndims = in_dim.size();

    std::vector<int> axis(axis_temp);
    std::vector<int> in_offset(ndims, 1);
    std::vector<int> out_offset(ndims, 1);

    for (size_t i = 0; i < axis.size(); i++) {
      axis[axis_temp[i]] = i;
    }

    for (int i = ndims - 2; i >= 0; i--) {
      in_offset[i] = in_offset[i + 1] * in_dim[i + 1];
      out_offset[i] = out_offset[i + 1] * out_dim[i + 1];
    }

    int block = 512;
    int grid = (data_size + block - 1) / block;
    transpose_kernel << grid, block >> (data_size, in_data, out_data, ndims,
                                        axis, in_offset, out_offset);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(transpose, ops::TransposeCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(transpose_grad, ops::TransposeGradCUDAKernel<float>);
