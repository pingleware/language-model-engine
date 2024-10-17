// model_inference.h
#pragma once
#include <onnxruntime/onnxruntime_cxx_api.h>

class ModelInference {
public:
    ModelInference(const std::string& model_path);
    std::vector<float> run_inference(const std::vector<float>& input_data);
    
private:
    Ort::Env env;
    Ort::Session session;
};