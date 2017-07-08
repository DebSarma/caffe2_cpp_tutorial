#ifndef AFFINE_TRANSFORM_OP_H
#define AFFINE_TRANSFORM_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class AffineTransformOp final : public Operator<Context> {
 public:
  AffineTransformOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int inverse_;
};

template <typename T, class Context>
class AffineTransformGradientOp final : public Operator<Context> {
 public:
  AffineTransformGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      inverse_(OperatorBase::GetSingleArgument<int>("inverse", 0)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int inverse_;
};

} // namespace caffe2

#endif // AFFINE_TRANSFORM_OP_H
