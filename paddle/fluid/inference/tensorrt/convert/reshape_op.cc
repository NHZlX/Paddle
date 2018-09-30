/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * MulOp, IMatrixMultiplyLayer in TRT. This Layer doesn't has weights.
 */
class ReshapeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims dims_input = input->getDimensions();

    const std::vector<int> shape =
        boost::get<std::vector<int>>(op_desc.GetAttr("shape"));
    
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *const_cast<nvinfer1::ITensor*>(input))
    PADDLE_ENFORCE(layer != nullptr);

    nvinfer1::Dims reshape_dims;
    reshape_dims.nbDims = int(shape.size() - 1);
    for (int32_t i = 0; i < reshape_dims.nbDims; ++i) {
      reshape_dims.d[i] = shape[i];
      reshape_dims.type[i] = dims_input.type[i];
    }
    layer->setReshapeDimensions(reshape_dims);
    auto output_name = op_desc.Output("Out")[0];
    nvinfer1::Dims dims_out = layer->getOutput(0)->getDimensions();

    int input_num = 0;
    int output_num = 0;
    for (int i = 0; i < dims_input.nbDims; i++) {
      input_num += dims_input.d[i];
      output_num += dims_out.d[i];
    }
    PADDLE_ENFORCE(input_num, output_num);

    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(reshape, ReshapeOpConverter);
