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

// DEFINE_string(dirname, "./checkouts2", "Directory of the inference model.");
// DEFINE_string(dirname, "./resnet56_batchnorm",
// DEFINE_string(dirname, "./map_model_part2",
//DEFINE_string(dirname, "./checkouts_googlenet_softmax1",
//DEFINE_string(dirname, "./checkouts_resnet50_softmax",
//DEFINE_string(dirname, "./checkouts6",
//DEFINE_string(dirname, "./checkouts_resnet50_softmax",
//DEFINE_string(dirname, "./checkouts_mobilenet_softmax",
//DEFINE_string(dirname, "./checkouts6",
//DEFINE_string(dirname, "./checkouts_mobilenet_softmax",
//DEFINE_string(dirname, "./checkouts_googlenet_softmax4",
//DEFINE_string(dirname, "./checkouts8",
//DEFINE_string(dirname, "./checkouts_vgg19",
//DEFINE_string(dirname, "./ssd_mobilenet_v1_pascalvoc",
//DEFINE_string(dirname, "./checkouts_resnext152",
DEFINE_string(dirname, "./checkouts8",
              "Directory of the inference model.");

TensorRTConfig GetConfig() {
  TensorRTConfig config;
  config.model_dir = FLAGS_dirname;
  // config.prog_file=FLAGS_dirname + "/__model__";
  // config.param_file=FLAGS_dirname + "/__params__";
  config.use_gpu = true;
  config.fraction_of_gpu_memory = 0.7;
  config.device = 0;
  config.max_batch_size=3;
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


void test_trt(int batch_size){
  TensorRTConfig config = GetConfig();
  config.max_batch_size = batch_size;
  auto predictor =
      CreatePaddlePredictor<TensorRTConfig,
                            PaddleEngineKind::kAutoMixedTensorRT>(config);
  //float data[batch_size * 3 * 28 * 28] = {1.0f};
  int height = 3;
  int width = 3;
  //int height = 224;
  //int width = 224;
  float data[batch_size * 3 * height * width] = {1.0f};
  // float data[1 * 3 * 3 * 3] = {1.0f};
  PaddleTensor tensor;
  tensor.name = "input_0";
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  //tensor.shape = std::vector<int>({batch_size, 1, 50, 1});
  //tensor.shape = std::vector<int>({1, 3, 3, 3});
  tensor.data =
     PaddleBuf(static_cast<void*>(data), sizeof(float) * (batch_size * 3 * height * width));
     //PaddleBuf(static_cast<void*>(data), sizeof(float) * (batch_size * 1 * 50 * 1));
      //PaddleBuf(static_cast<void*>(data), sizeof(float) * (3 * 3 * 3));
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
  predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
  std::cout << "start predict123:" << std::endl;
  auto time1 = time(); 
  for(int i = 0; i < 100; i++)  {
    predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
  } 

  auto time2 = time(); 

  std::cout <<"batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / 100.0 << "ms" << std::endl;
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}

TEST(alexnet, tensrrt) {
  std::vector<std::string> models_dir = {"./"};
  for (int i = 1; i <= 1; i++) {
    test_trt(i);
  }
}


}  // namespace paddle
