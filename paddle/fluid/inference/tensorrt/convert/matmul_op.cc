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
class MatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid matmul op to tensorrt mul layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);
    nvinfer1::Dims dims_input1 = input1->getDimensions();
    nvinfer1::Dims dims_input2 = input2->getDimensions();
    
    PADDLE_ENFORCE(dims_input1.nbDims == dims_input2.nbDims);
    // Both the input1 and input2 do not need transpose.
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *const_cast<nvinfer1::ITensor*>(input1), false,
        *const_cast<nvinfer1::ITensor*>(input2), false);

    const bool transpose_X =
        boost::get<bool>(op_desc.GetAttr("transpose_X"));

    const bool transpose_Y =
        boost::get<bool>(op_desc.GetAttr("transpose_Y"));

    if (transpose_X) {
    	layer->setTranspose(0, true);
    }

    if (transpose_Y) {
    	layer->setTranspose(1, true);
    }
    
    auto output_name = op_desc.Output("Out")[0];
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

REGISTER_TRT_OP_CONVERTER(matmul, MatMulOpConverter);
