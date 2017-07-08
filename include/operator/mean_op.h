#ifndef MEAN_OP_H
#define MEAN_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class MeanOp final : public Operator<Context> {
 public:
  MeanOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      offset_(OperatorBase::GetSingleArgument<int>("offset", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int offset_;
};

template <typename T, class Context>
class MeanGradientOp final : public Operator<Context> {
 public:
  MeanGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(OperatorBase::GetSingleArgument<int>("offset", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int offset_;
};

} // namespace caffe2

#endif // MEAN_OP_H
