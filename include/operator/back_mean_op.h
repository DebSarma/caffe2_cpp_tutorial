#ifndef BACK_MEAN_OP_H
#define BACK_MEAN_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class BackMeanOp final : public Operator<Context> {
 public:
  BackMeanOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <typename T, class Context>
class BackMeanGradientOp final : public Operator<Context> {
 public:
  BackMeanGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // BACK_MEAN_OP_H
