#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include <numeric>
#include <chrono>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gtest/gtest.h>
#include <cmath>

namespace paddle {
DEFINE_string(dirname, "", "Directory of the inference model.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

TensorRTConfig GetConfig() {
  TensorRTConfig config;
  config.prog_file = FLAGS_dirname + "/__model__";
  config.param_file = FLAGS_dirname + "/__param__";
  config.use_gpu = true;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  // Specific the variable's name of each input.
  config.specify_input_name = true;
  config.max_batch_size=1;
  return config;
}

bool test_map_cnn(int batch_size, int repeat) {
  TensorRTConfig config = GetConfig();
  config.max_batch_size = batch_size;
                                        // int64  int64.   fp32, int64, fp32, fp32
  std::vector<std::string> input_names({"src_word", "src_pos", "src_slf_attn_bias", "trg_word", "init_score", "trg_src_attn_bias"});

  std::vector<std::vector<int>> shapes({{batch_size, 256, 1},
                                        {batch_size, 256, 1},
                                        {batch_size, 8, 256, 256},
                                        {batch_size, 256, 1},
                                        {batch_size, 1},
                                        {batch_size, 8, 256, 256}});

  float *input_data[shapes.size()];
  int index = 0;
  for (auto& shape : shapes) {
    size_t data_size = (accumulate(shape.begin(), shape.end(), 1,
                                [](int a, int b) { return a * b; }));
    input_data[index] = new float[data_size];
    memset(input_data[index], 100, sizeof(float) * data_size);
    index += 1;
  }

  std::vector<PaddleTensor> inputs;
  LOG(INFO) << "inputs  ";

  index = 0;
  for (auto& shape : shapes) {
    
    // For simplicity, max_batch as the batch_size
    //shape.insert(shape.begin(), max_batch);
    // shape.insert(shape.begin(), 1);
    PaddleTensor feature;
    feature.name = input_names[index];
    feature.shape = shape;
    // feature.lod = std::vector<std::vector<size_t>>();
    size_t data_size = (sizeof(float) *  
                          accumulate(shape.begin(), shape.end(), 1,
                                          [](int a, int b) { return a * b; }));

    feature.data = PaddleBuf(static_cast<void *>(input_data[index]), data_size);
    feature.dtype = PaddleDType::FLOAT32;
    inputs.emplace_back(feature);
    index += 1;
  }
  LOG(INFO) << "predcit:  ";

  // warm-up
  auto predictor =
      CreatePaddlePredictor<TensorRTConfig,
                            PaddleEngineKind::kAutoMixedTensorRT>(config);
  // { batch begin
  std::vector<PaddleTensor> outputs;
  CHECK(predictor->Run(inputs, &outputs, batch_size));

  auto time1 = time(); 
  for(int i = 0; i < repeat; i++)  {
  	CHECK(predictor->Run(inputs, &outputs, batch_size));
  }
  auto time2 = time(); 
  std::cout <<"batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / repeat << "ms" << std::endl;
  float* data_o = static_cast<float*>(outputs[0].data.data());
 
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
    if (fabs(data_o[j] - 0.592781) >= 1e-5) return false;
  }
  return true;
}

TEST(map_cnn, tensorrt) {
 for(int i = 0; i < 1; i++) {
   ASSERT_TRUE(test_map_cnn(1 << i, 1));   
 }
}
}
