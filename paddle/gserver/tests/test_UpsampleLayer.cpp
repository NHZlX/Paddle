/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "LayerGradUtil.h"
#include "paddle/math/MathUtils.h"
#include "paddle/testing/TestUtil.h"
#include "iostream"

using namespace paddle;

void doOneUpsampleTest(MatrixPtr& input1,
                       MatrixPtr& input2,
                       MatrixPtr& input3,
                       bool use_gpu,
                       MatrixPtr& result) {
    TestConfig config;
    config.biasSize = 0;
    config.layerConfig.set_type("upsample");
    config.inputDefs.push_back({INPUT_DATA, "layer_0", 4, 0});
    LayerInputConfig* input = config.layerConfig.add_inputs();
    config.inputDefs.push_back({INPUT_DATA, "layer_1", 4, 0});
    config.layerConfig.add_inputs();
    UpsampleConfig* upsampleConfig = input->mutable_upsample_conf();
    upsampleConfig->set_scale(2);
    ImageConfig* imageConfig = upsampleConfig->mutable_image_conf();
    imageConfig->set_channels(1);
    imageConfig->set_img_size(2);
    imageConfig->set_img_size_y(2);
    config.layerConfig.set_size(2 * 4 * 4);

    config.layerConfig.set_name("upsample");

    std::vector<DataLayerPtr> dataLayers;
    LayerMap layerMap;
    vector<Argument> datas;
    initDataLayer(config,
                &dataLayers,
                &datas,
                &layerMap,
                "upsample",
                1,
                false,
                use_gpu);

    dataLayers[0]->getOutputValue()->copyFrom(*input1);
    dataLayers[1]->getOutputValue()->copyFrom(*input2);

    FLAGS_use_gpu = use_gpu;
    std::vector<ParameterPtr> parameters;
    LayerPtr upsampleLayer;
    initTestLayer(config, &layerMap, &parameters, &upsampleLayer);
    upsampleLayer->forward(PASS_GC);
    checkMatrixEqual(upsampleLayer->getOutput("").value,
                   result);
    upsampleLayer->getOutputGrad()->copyFrom(*input3);
    upsampleLayer->backward();
    dataLayers[0]->getOutputGrad()->print(std::cout);
}

TEST(Layer, upsampleLayerFwd) {
  bool useGpu = true;
  MatrixPtr input1;
  MatrixPtr input2;
  MatrixPtr input3;
  MatrixPtr result;

  real inputData1[] = {0.1, 0.2, 0.3, 0.4};
  real inputData2[] = {1, 2, 2, 4};
  real inputData3[] = {1, 0, 1.9, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  real resultData[] = {0, 0.1, 0.3, 0.0, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


  input1 = Matrix::create(1, 4, false, useGpu);
  input2 = Matrix::create(1, 4, false, useGpu);
  result = Matrix::create(1, 4*4, false, useGpu);
  input3 = Matrix::create(1, 4*4, false, useGpu);
  input1->setData(inputData1);
  input2->setData(inputData2);
  input3->setData(inputData3);
  result->setData(resultData);
  doOneUpsampleTest(
      input1, input2, input3, useGpu, result);
}
