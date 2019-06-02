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

#include "paddle/fluid/inference/anakin/convert/resize.h"
#include <algorithm>
#include <map>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

void ResizeOpConverter::operator()(const framework::proto::OpDesc &op,
                                   const framework::Scope &scope,
                                   bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  auto input_name = op_desc.Input("X").front();
  auto output_name = op_desc.Output("Out").front();

  int out_h = boost::get<int>(op_desc.GetAttr("out_h"));
  int out_w = boost::get<int>(op_desc.GetAttr("out_w"));
  float scale = boost::get<float>(op_desc.GetAttr("scale"));
  // auto resize_method =
  // boost::get<std::string>(op_desc.GetAttr("interp_method"));
  // bool align_corners = boost::get<bool>(op_desc.GetAttr("align_corners"));
  // int align_mode = boost::get<int>(op_desc.GetAttr("align_mode"));
  engine_->AddOp(op_name, "Resize", {input_name}, {output_name});

  float scale_h, scale_w;
  if (scale > 0) {
    scale_h = scale;
    scale_w = scale;
    engine_->AddOpAttr(op_name, "height_scale", scale_h);
    engine_->AddOpAttr(op_name, "width_scale", scale_w);
  } else {
    engine_->AddOpAttr(op_name, "out_height", out_h);
    engine_->AddOpAttr(op_name, "out_width", out_w);
  }
  engine_->AddOpAttr(op_name, "method", std::string("BILINEAR_ALIGN"));
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(bilinear_interp, ResizeOpConverter);
