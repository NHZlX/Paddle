#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include <numeric>
#include <chrono>
#include <gtest/gtest.h>

namespace paddle {
DEFINE_string(dirname, "./", "Directory of the inference model.");

NativeConfig GetConfig() {
  NativeConfig config;
  config.model_dir = FLAGS_dirname;
  // config.prog_file=FLAGS_dirname + "/model";
  //config.param_file=FLAGS_dirname + "/params";
  LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.15;
  //config.use_gpu = true;
  config.use_gpu = true;
  config.device = 0;
  config.specify_input_name = true;
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

void test_map_cnn(int batch_size) {
  NativeConfig config = GetConfig();
  std::vector<std::string> input_names({"pixel", "init_ids", "init_scores", "position_encoding"});

  std::vector<std::vector<int>> shapes({{batch_size, 1, 100, 200},
                                        {batch_size, 1},
                                        {batch_size, 1},
                                        {batch_size, 33, 10, 23}});

  size_t pixel_size = (accumulate(shapes[0].begin(), shapes[0].end(), 1,
                                [](int a, int b) { return a * b; }));
  float *pixel = new float[pixel_size];
  memset(pixel, 1, sizeof(float) * pixel_size);
  PaddleTensor pixel_t;
  pixel_t.name = input_names[0];
  pixel_t.shape = shapes[0];
  pixel_t.data = PaddleBuf(static_cast<void *>(pixel), pixel_size * sizeof(float));
  pixel_t.dtype = PaddleDType::FLOAT32;


  std::vector<std::vector<size_t>> lod_s = {{0, 1}, {0, 1}};
  size_t ids_size = (accumulate(shapes[1].begin(), shapes[1].end(), 1,
                                [](int a, int b) { return a * b; }));
  int64_t *init_ids = new int64_t[ids_size];
  memset(init_ids, 0, sizeof(int64_t) * ids_size);
  PaddleTensor ids_t;
  ids_t.name = input_names[1];
  ids_t.shape = shapes[1];
  ids_t.data = PaddleBuf(static_cast<void *>(init_ids), ids_size * sizeof(int64_t));
  ids_t.dtype = PaddleDType::INT64;
  ids_t.lod = lod_s;

  size_t scores_size = (accumulate(shapes[2].begin(), shapes[2].end(), 1,
                                [](int a, int b) { return a * b; }));
  float *init_scores = new float[scores_size];
  memset(init_scores, 1, sizeof(float) * scores_size);

  PaddleTensor scores_t;
  scores_t.name = input_names[2];
  scores_t.shape = shapes[2];
  scores_t.data = PaddleBuf(static_cast<void *>(init_scores), scores_size * sizeof(float));
  scores_t.dtype = PaddleDType::FLOAT32;
  scores_t.lod = lod_s;
  

  size_t position_size = (accumulate(shapes[3].begin(), shapes[3].end(), 1,
                                [](int a, int b) { return a * b; }));
  float *position_encoding = new float[position_size];
  memset(position_encoding, 1, sizeof(float) * position_size);
  PaddleTensor pos_t;
  pos_t.name = input_names[3];
  pos_t.shape = shapes[3];
  pos_t.data = PaddleBuf(static_cast<void *>(position_encoding), position_size * sizeof(float));
  pos_t.dtype = PaddleDType::FLOAT32;
  
  std::vector<PaddleTensor> inputs;

  inputs.push_back(pixel_t);
  inputs.push_back(ids_t);
  inputs.push_back(scores_t);
  inputs.push_back(pos_t);
 

  LOG(INFO) << "predcit:  ";

  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  // { batch begin
  std::vector<PaddleTensor> outputs;
  auto time1 = time(); 
  for(int i = 0; i < 100; i++)  {
  	CHECK(predictor->Run(inputs, &outputs));
  }
  auto time2 = time(); 
  std::cout <<"batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / 100.0 << "ms" << std::endl;
  for (size_t i = 0; i < outputs.size(); i++) {
     std::cout << "Num: " << i << std::endl;
     float* data_o = static_cast<float*>(outputs[i].data.data());
     for (size_t j = 0; j < outputs[i].data.length() / sizeof(float); ++j) {
        LOG(INFO) << "output[" << j << "]: " << data_o[j];
     }
  }
  // } batch end
}

TEST(map_cnn, naive) {
 for(int i = 0; i <= 0; i++) {
   test_map_cnn(1 << i);    
 }
}
}
