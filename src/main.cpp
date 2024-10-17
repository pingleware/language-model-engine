#include "model_inference.h"
#include "opengl_renderer.h"

int main() {
    // Initialize model inference
    ModelInference model_inference("models/model.onnx");

    // Input data (e.g., from user input or static data)
    std::vector<float> input_data = {/*... populate input data ...*/};

    // Run inference
    std::vector<float> output = model_inference.run_inference(input_data);

    // Render the output using OpenGL
    OpenGLRenderer renderer;
    renderer.render(output);

    return 0;
}