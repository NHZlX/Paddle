#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
#include <set>
#include <string.h>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <fstream>
#include <gflags/gflags.h>
#include <chrono>

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

DEFINE_string(dirname, "./", "Directory of the inference model.");
std::unique_ptr<paddle::PaddlePredictor> fluid_predictor;

void convert_output(const std::vector<paddle::PaddleTensor> &tensors,
                    std::vector<std::vector<float> > &datas,
                    std::vector<std::vector<int> > &shapes) {
    // use reference to avoid double free
    for (auto &t : tensors) {
        shapes.push_back(t.shape);
        const size_t num_elements = t.data.length() / sizeof(float);
        float *tdata = static_cast<float*>(t.data.data());
        std::vector<float> data(num_elements, 0);
        std::copy(tdata, tdata + num_elements, data.data());
        datas.push_back(data);
    }
}

std::string fluid_predict() {
    int height = 100;
    int width = 200;
    std::vector<paddle::PaddleTensor> input_tensors;  
    std::vector<paddle::PaddleTensor> output_tensors;
    //image tensor
    paddle::PaddleTensor image_tensor;
    std::vector<int> image_shape;
    image_shape.push_back(1);
    image_shape.push_back(1);
    image_shape.push_back(100);
    image_shape.push_back(200);
    std::vector<float> image_data;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
                //image_data.push_back((float)());
                image_data.push_back((float)(1.0));
        }
    }
	
	std::cerr << "image size is " << image_data.size() << std::endl;
    image_tensor.shape = image_shape;
    image_tensor.dtype = paddle::PaddleDType::FLOAT32;

    image_tensor.data.Resize(sizeof(float) * 100 * 200);
    std::copy(image_data.begin(), image_data.end(), static_cast<float*>(image_tensor.data.data()));
    image_tensor.name = "pixel";

    paddle::PaddleTensor init_ids_tensor;
    std::vector<int> ids_shape;
    ids_shape.push_back(1);
    ids_shape.push_back(1);
    std::vector<int64_t> init_ids;
    init_ids.push_back(0);
    init_ids_tensor.shape = ids_shape;
    init_ids_tensor.dtype = paddle::PaddleDType::INT64;
    //init_ids_tensor.data = init_ids;
    init_ids_tensor.data.Resize(sizeof(int64_t) * 1);
    init_ids_tensor.name = "init_ids";
    std::copy(init_ids.begin(), init_ids.end(), static_cast<int64_t*>(init_ids_tensor.data.data()));
    std::vector<size_t> lod_1;
    lod_1.push_back(0);
    lod_1.push_back(1);
    std::vector<size_t> lod_2;
    lod_2.push_back(0);
    lod_2.push_back(1);
    std::vector<std::vector<size_t>> lod;
    lod.push_back(lod_1);
    lod.push_back(lod_2);
    init_ids_tensor.lod = lod;
    
    //init scores
    paddle::PaddleTensor init_scores_tensor;
    std::vector<int> scores_shape;
    scores_shape.push_back(1);
    scores_shape.push_back(1);
    std::vector<float> init_scores;
    init_scores.push_back(1.0);
    init_scores_tensor.shape = scores_shape;
    init_scores_tensor.dtype = paddle::PaddleDType::FLOAT32;
    //init_scores_tensor.data = init_scores;
	init_scores_tensor.data.Resize(sizeof(float) * 1);
	std::copy(init_scores.begin(), init_scores.end(), static_cast<float*>(init_scores_tensor.data.data()));
    init_scores_tensor.name = "init_scores";
    init_scores_tensor.lod = lod;
    //position encoding
    paddle::PaddleTensor position_encoding_tensor;
    std::vector<int> position_encoding_shape;
    position_encoding_shape.push_back(1);
    position_encoding_shape.push_back(33);
    position_encoding_shape.push_back(10);
    position_encoding_shape.push_back(23);
    std::vector<int> pos_data;
    for (int i = 0; i < 10; i++) {
        for (int row = 0; row < 10; row++) {
            for (int col = 0; col < 23; col++) {
                if (i == row) {
                    pos_data.push_back(1);
                } else {
                    pos_data.push_back(0);
                }
                
            }
        }
    }
    for (int i = 0; i < 23; i++) {
        for (int row = 0; row < 10; row++) {
            for (int col = 0; col < 23; col++) {
                if (i == col) {
                    pos_data.push_back(1);
                } else {
                    pos_data.push_back(0);
                }
                
            }
        }
    }
    position_encoding_tensor.name = "position_encoding";
    position_encoding_tensor.shape = position_encoding_shape;
    position_encoding_tensor.dtype = paddle::PaddleDType::FLOAT32;
    //position_encoding_tensor.data = pos_data;
	position_encoding_tensor.data.Resize(sizeof(float) * 33 * 10 * 23);
	std::copy(pos_data.begin(), pos_data.end(), static_cast<float*>(position_encoding_tensor.data.data()));
    input_tensors.push_back(image_tensor);
    input_tensors.push_back(init_ids_tensor);
    input_tensors.push_back(init_scores_tensor);
    input_tensors.push_back(position_encoding_tensor);
	std::cerr << "before prediction\n";
    fluid_predictor->Run(input_tensors, &output_tensors);
    auto time1 = time(); 
    for(int i = 0; i < 1000; i++)  {
      fluid_predictor->Run(input_tensors, &output_tensors);
    }
    auto time2 = time(); 
    std::cout <<"batch: " << 1 << " predict cost: " << time_diff(time1, time2) / 100.0 << "ms" << std::endl;
	std::vector<std::vector<float> > output_data;
	std::vector<std::vector<int> > output_shapes;
	convert_output(output_tensors, output_data, output_shapes);
	
	std::string plate_str = "";
	for (float k : output_data[0]) {
		std::cerr << "text\t" << k << std::endl;
		if (k == 0 || k == 1 || k == 2) {
			continue;
		}
		//plate_str += table_dict[k];
	}
	
	for (float k : output_data[1]) {
		std::cerr << "scores\t" << k << std::endl;
	}
    return plate_str;
}
    
int main(int argc, char** argv) {
    //1. init image recognition model
    paddle::NativeConfig paddle_config;
    float fraction_of_gpu_memory = 0.3f;
    bool use_gpu = false;
    std::string prog_file =  "./InceptionV3_Model/model";
    std::string param_file = "./InceptionV3_Model/params";
    std::cout << prog_file << std::endl;
    
    paddle_config.fraction_of_gpu_memory = fraction_of_gpu_memory;
    paddle_config.use_gpu = use_gpu;
    paddle_config.device=0;
    paddle_config.prog_file = prog_file;
    paddle_config.param_file = param_file;
    paddle_config.specify_input_name = true;
    fluid_predictor =
      paddle::CreatePaddlePredictor<paddle::NativeConfig, paddle::PaddleEngineKind::kNative>(paddle_config);


    double total_time_cost = 0.0f;
    int total_number = 1;

     timeval start_time;
	gettimeofday(&start_time, NULL);        
	std::string predict_str = fluid_predict();
	timeval end_time;
	gettimeofday(&end_time, NULL);
	double time_cost = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
	std::cout << "predict time is [" << time_cost << "]" << std::endl;
	total_time_cost += time_cost;
    total_number += 1;
    std::cout << "avg time cost is [" << total_time_cost / total_number  << "]" << std::endl;
    
    return 0;
}
