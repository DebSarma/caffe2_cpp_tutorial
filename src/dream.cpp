#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/utils/proto_utils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "util/models.h"
#include "util/print.h"
#include "util/image.h"
#include "util/cuda.h"
#include "util/build.h"
#include "util/net.h"
#include "util/math.h"
#include "res/imagenet_classes.h"
#include "operator/operator_cout.h"


CAFFE2_DEFINE_string(model, "googlenet", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "inception_3b/5x5_reduce", "Name of the layer on which to split the model.");
CAFFE2_DEFINE_int(offset, 2, "The first channel to run.");
CAFFE2_DEFINE_int(batch, 1, "The number of channels to process in parallel.");
CAFFE2_DEFINE_int(size, 400, "The goal image size.");

CAFFE2_DEFINE_int(round_count, 10, "The number of different scales.");
CAFFE2_DEFINE_int(round_steps, 10, "The amount of iterations per scale.");
CAFFE2_DEFINE_int(percent_incr, 40, "Percent increase per round.");
CAFFE2_DEFINE_int(initial, 0, "The of initial value.");
CAFFE2_DEFINE_double(learning_rate, 1, "Learning rate.");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");

namespace caffe2 {

void AddSuperNaive(NetDef &init_model, NetDef &predict_model, int label_index) {
  // add gradients
  auto &output = predict_model.external_output(0);
  add_xent_ops(predict_model, output);
  add_gradient_ops(predict_model);
  add_iter_lr_ops(init_model, predict_model, FLAGS_learning_rate);

  // add dream operator
  add_constant_fill_int32_op(init_model, { 1 }, label_index, "label");
  predict_model.add_external_input("label");
  add_constant_fill_float_op(init_model, { 1 }, 1.0, "one");
  predict_model.add_external_input("one");
  auto &input = predict_model.external_input(0);
  add_weighted_sum_op(predict_model, { input, "one", input + "_grad", "lr" }, input);
}

void AddNaive(NetDef &init_model, NetDef &predict_model, int size) {
  auto &input = predict_model.external_input(0);
  auto &output = predict_model.external_output(0);

  // initialize input data
  add_uniform_fill_float_op(init_model, { FLAGS_batch, 3, size, size }, FLAGS_initial, FLAGS_initial + 1, input);

  // add reduce mean as score
  add_back_mean_op(predict_model, output, "back1");
  add_back_mean_op(predict_model, "back1", "back2");
  add_diagonal_op(predict_model, "back2", "diagonal", { 0, FLAGS_offset });
  add_averaged_loss(predict_model, "diagonal", "score");
  add_constant_fill_with_op(predict_model, 1.0, "score", "score_grad");

  // add back prop
  add_gradient_ops(predict_model);

  // scale gradient
  add_mean_stdev_op(predict_model, input + "_grad", "_", input + "_grad_stdev");
  add_constant_fill_with_op(predict_model, 0.0, input + "_grad_stdev", "zero");
  add_scale_op(predict_model, input + "_grad_stdev", input + "_grad_stdev", 1 / FLAGS_learning_rate);
  add_affine_transform_op(predict_model, input + "_grad", "zero", input + "_grad_stdev", input + "_grad", true);

  // apply gradient to input data
  add_constant_fill_float_op(init_model, { 1 }, 1.0, "one");
  predict_model.add_external_input("one");
  add_weighted_sum_op(predict_model, { input, "one", input + "_grad", "one" }, input);

  // scale data to image
  add_mean_stdev_op(predict_model, input, input + "_mean", input + "_stdev");
  add_affine_transform_op(predict_model, input, input + "_mean", input + "_stdev", "image", true);
  add_scale_op(predict_model, "image", "image", 25.5);
  add_clip_op(predict_model, "image", "image", -128, 128);
}

void run() {
  std::cout << std::endl;
  std::cout << "## Deep Dream Example ##" << std::endl;
  std::cout << std::endl;

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair: model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  // if (!FLAGS_label.size()) {
  //   std::cerr << "specify a label name using --label <name>" << std::endl;
  //   return;
  // }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "layer: " << FLAGS_layer << std::endl;
  std::cout << "offset: " << FLAGS_offset << std::endl;
  std::cout << "batch: " << FLAGS_batch << std::endl;
  std::cout << "size: " << FLAGS_size << std::endl;

  std::cout << "round_count: " << FLAGS_round_count << std::endl;
  std::cout << "round_steps: " << FLAGS_round_steps << std::endl;
  std::cout << "percent_incr: " << FLAGS_percent_incr << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "force_cpu: " << (FLAGS_force_cpu ? "true" : "false") << std::endl;

  if (!FLAGS_force_cpu) setupCUDA();

  // auto label_index = -1;
  // for (int i = 0; i < 1000; i++) {
  //   if (!strcmp(FLAGS_label.c_str(), imagenet_classes[i])) {
  //     label_index = i;
  //   }
  // }
  // if (label_index < 0) {
  //   for (int i = 0; i < 1000; i++) {
  //     std::cout << "  " << imagenet_classes[i] << std::endl;
  //   }
  //   LOG(FATAL) << "~ image class label not found: " << FLAGS_label;
  // }

  std::cout << std::endl;

  std::cout << "loading model.." << std::endl;
  clock_t load_time = 0;
  NetDef base_init_model, base_predict_model;

  // check if model present
  CAFFE_ENFORCE(ensureModel(FLAGS_model), "model ", FLAGS_model, " not found");
  std::string init_filename = "res/" + FLAGS_model + "_init_net.pb";
  std::string predict_filename = "res/" + FLAGS_model + "_predict_net.pb";

  // read model files
  load_time -= clock();
  CAFFE_ENFORCE(ReadProtoFromFile(init_filename.c_str(), &base_init_model));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_filename.c_str(), &base_predict_model));
  load_time += clock();

  // std::cout << join_net(base_init_model);
  // std::cout << join_net(base_predict_model);

  // std::cout << collect_layers(base_predict_model, "inception_3a/5x5") << std::endl;

  CheckLayerAvailable(base_predict_model, FLAGS_layer);

  // extract dream model
  NetDef dream_init_model, dream_predict_model, second_init_model, second_predict_model;
  SplitModel(base_init_model, base_predict_model, FLAGS_layer, dream_init_model, dream_predict_model, second_init_model, second_predict_model, FLAGS_force_cpu, false);

  // add dream operators
  auto image_size = FLAGS_size;
  for (int i = 1; i < FLAGS_round_count; i++) {
    image_size = image_size * 100 / (100 + FLAGS_percent_incr);
  }
  AddNaive(dream_init_model, dream_predict_model, image_size);

  // std::cout << join_net(dream_init_model);
  // std::cout << join_net(dream_predict_model);

  // set model to use CUDA
  if (!FLAGS_force_cpu) {
    set_device_cuda_model(dream_init_model);
    set_device_cuda_model(dream_predict_model);
  }

  std::cout << "running model.." << std::endl;
  clock_t dream_time = 0;
  Workspace workspace;

  // setup workspace
  auto init_net = CreateNet(dream_init_model, &workspace);
  auto predict_net = CreateNet(dream_predict_model, &workspace);
  init_net->Run();

  auto &input_name = dream_predict_model.external_input(0);

  // read image as tensor
  // auto input = readImageTensor(FLAGS_image_file, FLAGS_size / 10);
  // set_tensor_blob(*workspace.GetBlob(input_name), input);
  // showImageTensor(input, 0);

  // run predictor
  for (auto step = 0; image_size < FLAGS_size;) {

    // scale up image tiny bit
    image_size = std::min(image_size * (100 + FLAGS_percent_incr) / 100, FLAGS_size);
    auto data = get_tensor_blob(*workspace.GetBlob("data"));
    auto scaled = scaleImageTensor(data, image_size, image_size);
    set_tensor_blob(*workspace.GetBlob("data"), scaled);

    for (int i = 0; i < FLAGS_round_steps; i++, step++) {
      dream_time -= clock();
      predict_net->Run();
      dream_time += clock();

      if (!step) {
        auto depth = get_tensor_blob(*workspace.GetBlob(FLAGS_layer)).dim(1);
        std::cout << "channel depth: " << depth << std::endl;
        // print(*workspace.GetBlob(FLAGS_layer), FLAGS_layer);
      }
    }

    auto score = get_tensor_blob(*workspace.GetBlob("score")).data<float>()[0];
    std::cout << "step: " << step << "  score: " << score << "  size: " << image_size << std::endl;

    // show current images
    auto image = get_tensor_blob(*workspace.GetBlob("image"));
    showImageTensor(image, FLAGS_size / 2, FLAGS_size / 2);
  }

  auto image = get_tensor_blob(*workspace.GetBlob("image"));
  auto safe_layer = FLAGS_layer;
  std::replace(safe_layer.begin(), safe_layer.end(), '/', '_');
  writeImageTensor(image, "dream/" + safe_layer + "_" + std::to_string(FLAGS_offset));

  std::cout << std::endl;

  std::cout << std::setprecision(3)
    << "load: " << ((float)load_time / CLOCKS_PER_SEC)
    << "s  dream: " << ((float)dream_time / CLOCKS_PER_SEC) << "s"
    << std::endl;
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
