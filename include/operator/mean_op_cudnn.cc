#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "operator/mean_op.h"

namespace caffe2 {

template <typename T>
class CuDNNMeanOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNMeanOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        offset_(OperatorBase::GetSingleArgument<int>("offset", 1)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    get_mean_tensor(X, Y);
    return true;
  }

 protected:
  int offset_;
};


template <typename T>
class CuDNNMeanGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNMeanGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        offset_(OperatorBase::GetSingleArgument<int>("offset", 1)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    set_mean_tensor(dX, dY);
    return true;
  }

 protected:
  int offset_;
};

namespace {
REGISTER_CUDNN_OPERATOR(Mean, CuDNNMeanOp<float>);
REGISTER_CUDNN_OPERATOR(MeanGradient, CuDNNMeanGradientOp<float>);

}  // namespace

}  // namespace caffe2

#endif
