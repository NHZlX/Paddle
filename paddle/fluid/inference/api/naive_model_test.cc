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
#include <chrono>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

// DEFINE_string(dirname, "./checkouts3", "Directory of the inference model.");
DEFINE_string(dirname, "./checkouts_resnet50_softmax_prune",
              "Directory of the inference model.");

NativeConfig GetConfig() {
  NativeConfig config;
  config.model_dir = FLAGS_dirname;
  LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.15;
  config.use_gpu = true;
  config.device = 0;
  return config;
}

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

TEST(alexnet, tensrrt) {
  NativeConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<NativeConfig>(config);
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
  std::cout << "start predict:" << std::endl;
  predictor->Run(paddle_tensor_feeds, &outputs, 1);
  std::cout << "start predict123:" << std::endl;
  auto time1 = time(); 
  for(int i = 0; i < 1; i++)  {
    predictor->Run(paddle_tensor_feeds, &outputs, 1);
  } 

  auto time2 = time(); 

  std::cout << "predict cost: " << time_diff(time1, time2) / 1.0 << "ms" << std::endl;
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}

}  // namespace paddle
