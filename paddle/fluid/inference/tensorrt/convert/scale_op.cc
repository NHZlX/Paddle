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
 * ScaleOp, Y = a * X, a is a float num. 
 */
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const float scale =
        boost::get<float>(op_desc.GetAttr("scale"));

    platform::CPUPlace cpu_place;
    std::unique_ptr<framework::LoDTensor> weight_tensor(
        new framework::LoDTensor());
    weight_tensor->Resize(framework::make_ddim({1}));
    auto* weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    weight_data[0] = scale; 

    TensorRTEngine::Weight scale_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(weight_data),
        weight_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight shift_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *const_cast<nvinfer1::ITensor*>(input),
        nvinfer1::ScaleMode::kUNIFORM, shift_weights.get(), scale_weights.get(),
        power_weights.get());

    PADDLE_ENFORCE(layer != nullptr);

    engine_->weight_map[op_desc.Output("Out").front() + "_scale"] =
        std::move(weight_tensor);
    auto output_name = op_desc.Output("Out")[0];

    engine_->SetITensor(output_name, layer->getOutput(0));
    layer->setName(("scale (Output: " + output_name + ")").c_str());
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(scale, ScaleOpConverter);
