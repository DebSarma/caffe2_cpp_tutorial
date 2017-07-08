#include "operator/affine_transform_op.h"
#include "util/math.h"

namespace caffe2 {

template <>
bool AffineTransformOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& M = Input(1);
  auto& S = Input(2);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  get_affine_transform_tensor(X, M, S, *Y, inverse_);
  return true;
}

template <>
bool AffineTransformGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& M = Input(1);
  auto& S = Input(2);
  auto& dY = Input(3);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  get_affine_transform_tensor(dY, M, S, *dX, !inverse_);
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(AffineTransform, AffineTransformOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineTransformGradient, AffineTransformGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(AffineTransform)
  .NumInputs(3)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
The operator affine transforms values per batch item.
)DOC")
  .Arg("inverse", "apply inverse of affine transform")
  .Input(0, "input", "The input data as N-D Tensor<float>.")
  .Input(1, "mean", "The mean values as 1-D Tensor<float> of batch size.")
  .Input(2, "scale", "The scale values as 1-D Tensor<float> of batch size.")
  .Output(0, "output", "Scaled N-D Tensor<float>.");

OPERATOR_SCHEMA(AffineTransformGradient).NumInputs(4).NumOutputs(1).AllowInplace({{0, 0}});

class GetAffineTransformGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AffineTransform, GetAffineTransformGradient);

}  // namespace

}  // namespace caffe2
