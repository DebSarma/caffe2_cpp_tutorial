#ifdef WITH_CUDA

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "operator/affine_transform_op.h"
#include "util/math.h"

namespace caffe2 {

template <typename T>
class CuDNNAffineTransformOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNAffineTransformOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& M = Input(1);
    auto& S = Input(2);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    get_affine_transform_tensor(X, M, S, *Y, inverse_);
    return true;
  }

 protected:
  int inverse_;
};


template <typename T>
class CuDNNAffineTransformGradientOp final : public Operator<CUDAContext> {
 public:
  explicit CuDNNAffineTransformGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws),
        inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& M = Input(1);
    auto& S = Input(2);
    auto& dY = Input(3);
    auto* dX = Output(0);
    dX->ResizeLike(X);
    get_affine_transform_tensor(dY, M, S, *dX, !inverse_);
    return true;
  }

 protected:
  int inverse_;
};

namespace {

REGISTER_CUDNN_OPERATOR(AffineTransform, CuDNNAffineTransformOp<float>);
REGISTER_CUDNN_OPERATOR(AffineTransformGradient, CuDNNAffineTransformGradientOp<float>);

}  // namespace

}  // namespace caffe2

#endif
