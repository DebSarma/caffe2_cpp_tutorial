#include "operator/mean_op.h"
#include "util/math.h"

namespace caffe2 {

template <>
bool MeanOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  get_mean_tensor(&X, Y);
  return true;
}

template <>
bool MeanGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  set_mean_tensor(dX, &dY);
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Mean, MeanOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MeanGradient, MeanGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Mean)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
The operator takes the mean values overlast dimension.
)DOC")
  .Input(0, "input", "The input data as n-D Tensor<float>.")
  .Output(0, "output", "The mean in a (n-1)-D Tensor.");

OPERATOR_SCHEMA(MeanGradient).NumInputs(2).NumOutputs(1);

class GetMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Mean, GetMeanGradient);

}  // namespace

}  // namespace caffe2
