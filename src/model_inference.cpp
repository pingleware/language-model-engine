// model_inference.cpp
#include "model_inference.h"
#include <iostream>

ModelInference::ModelInference(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "LanguageModel"), 
      session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {}

std::vector<float> ModelInference::run_inference(const std::vector<float>& input_data) {
    // Define input and output for the model (this is simplified)
    // std::vector<int64_t> input_shape = {1, input_data.size()};
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};
    size_t input_tensor_size = input_data.size();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_tensor_size, input_shape.data(), input_shape.size());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_tensor_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // Run the model
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Extract output (assuming a simple vector output)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + input_tensor_size);
}