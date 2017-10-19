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

#include "UpsampleLayer.h"

namespace paddle {

REGISTER_LAYER(scaling, ScalingLayer);

bool UpsampleLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(inputLayers_.size(), 2U);
  CHECK_EQ(config_.inputs_size(), 1);
  const UpsampleConfig& conf = config_.inputs(0).upsample_conf();
  CHECK((conf.has_upsample_size() && conf.has_upsample_size_y()) ||
        (conf.has_scale()))
      << "scale or scale & scale_y are required else "
      << "upsample_size or upsample_size & upsample_size_y are required";
  if (conf.has_upsample_size()) {
    upsampleSize_ = conf.upsample_size();
    upsampleSizeY_ = upsampleSize_;
    if (conf.has_upsample_size_y()) {
      upsampleSizeY_ = conf.upsample_size_y();
    }
  } else {
    if (!conf.has_scale_y()) {
      scale_ = scaleY_ = conf.scale_y();
      CHECK_GT(scale_, 1);
    } else {
      scale_ = conf.scale();
      scaleY_ = conf.scale_y();
    }
    padOutX_ = conf.pad_out_x();
    padOutY_ = conf.pad_out_y();
    CHECK(!padOutX_ || scale_)
        << "Output Height padding compensation requires scale_ == 2";
    CHECK(!padOutY_ || scaleY_)
        << "Output Width padding compensation requires scale_ == 2";
    upsampleSize_ = upsampleSizeY_ = -1;
  }
}
// getSize()

void UpsampleLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inputMatP = getInputValue(0);
  MatrixPtr maskMatP = getInputValue(1);

  size_t batchSize = inputMatP->getHeight();
  size_t dataDim = inputMatP->getWidth();

  CHECK_EQ(inputMatP->getWidth(), maskMatP->getWidth());
  CHECK_EQ(weightV->getHeight(), batchSize);

  resetOutput(batchSize, dataDim);

  MatrixPtr outV = getOutputValue();
  outV->upsampleForward(0, *inV1, *weightV);
}

void UpsampleLayer::backward(const UpdateCallback& callback) {
  MatrixPtr mask = getInputValue(1);
  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outGrad = getOutputGrad();
  outGrad->upsampleBackward();
}

}  // namespace paddle
