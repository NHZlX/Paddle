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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data file");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  VLOG(3) << "data size " << record.data.size();
  VLOG(3) << "data shape size " << record.shape.size();
  return record;
}

/*
 * Use the native and analysis fluid engine to inference the demo.
 * ocr, mobilenet and se_resnext50
 */

TensorRTConfig GetConfigTRT() {
  TensorRTConfig config;
  config.prog_file = FLAGS_infer_model + "/__model__";
  config.param_file = FLAGS_infer_model + "/__params__";
  //config.model_dir = FLAGS_dirname;
  config.use_gpu = true;
  config.fraction_of_gpu_memory = 0.3;
  config.device = 0;
  config.specify_input_name = true;
  config.max_batch_size = 3;
  config.minimum_subgraph_size=1;
  return config;
}

void TestVisualPrediction() {
  TensorRTConfig config1 = GetConfigTRT();
  config1.max_batch_size = FLAGS_batch_size;

  auto predictor1 =
      CreatePaddlePredictor<TensorRTConfig,
                            PaddleEngineKind::kAutoMixedTensorRT>(config1);

  // Only have single batch of data.
  std::string line;
  std::ifstream file(FLAGS_infer_data);
  std::getline(file, line);
  auto record = ProcessALine(line);
  file.close();

  // Inference.
  PaddleTensor input;
  input.shape = {1, 1, 48, 512};
  for (auto s : input.shape) {
   std::cout << "shape:" << s << std::endl;
  }
  std::cout << "size:" << record.data.size() * sizeof(float) << std::endl;
  input.data =
      PaddleBuf(record.data.data(), (1 * 48 * 512) * sizeof(float));
  input.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> outputs_slots;
  for (int i = 0; i < FLAGS_repeat; i++) {
    predictor1->Run({input}, &outputs_slots, FLAGS_batch_size);
  }

  VLOG(3) << "output.size " << outputs_slots.size();

  // run native as reference
  // print what are fused
  EXPECT_EQ(outputs_slots.size(), 1UL);
  auto &out = outputs_slots[0];
  size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                [](int a, int b) { return a * b; });
  ASSERT_GT(size, 0);
  int64_t *result = static_cast<int64_t *>(out.data.data());
  for (size_t i = 0; i < std::min(11UL, size); i++) {
    std::cout << result[i] << std::endl;
  }
}

TEST(Analyzer_vis, analysis) { TestVisualPrediction(); }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
