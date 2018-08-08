// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {

// DEFINE_string(dirname, "./checkouts3", "Directory of the inference model.");
DEFINE_string(dirname, "./checkouts_resnet50_softmax_prune",
              "Directory of the inference model.");

TensorRTConfig GetConfig() {
  TensorRTConfig config;
  config.model_dir = FLAGS_dirname;
  config.use_gpu = true;
  config.fraction_of_gpu_memory = 0.3;
  config.device = 0;
  return config;
}

TEST(alexnet, tensrrt) {
  TensorRTConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<TensorRTConfig,
                            PaddleEngineKind::kAutoMixedTensorRT>(config);
  float data[1 * 3 * 224 * 224] = {1.0f};
  // float data[1 * 3 * 3 * 3] = {1.0f};
  PaddleTensor tensor;
  tensor.name = "input_0";
  // tensor.shape = std::vector<int>({1, 3, 224, 224});
  tensor.shape = std::vector<int>({1, 3, 224, 224});
  tensor.data =
      PaddleBuf(static_cast<void*>(data), sizeof(float) * (3 * 224 * 224));
  tensor.dtype = PaddleDType::FLOAT32;

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  PaddleTensor tensor_out;
  tensor_out.name = "prob_out";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> outputs(1, tensor_out);
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs, 1));

  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}

}  // namespace paddle
USE_TRT_CONVERTER(elementwise_add_weight);
USE_TRT_CONVERTER(mul);
USE_TRT_CONVERTER(conv2d);
USE_TRT_CONVERTER(relu);
USE_TRT_CONVERTER(fc);
USE_TRT_CONVERTER(pool2d);
USE_TRT_CONVERTER(softmax);
