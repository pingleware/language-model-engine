#include <iostream>
#include <vector>
#include "model_inference.h"
#include "opengl_renderer.h"

int main() {
    // Initialize model inference
    ModelInference model_inference("models/model.onnx");

    // Define the dimensions of the input data
    const int batch_size = 1; // 1 sample
    const int input_size = 10; // Input size as expected by the model

    // Create and populate the input data
    std::vector<float> input_data(input_size);

    // Fill the input data with dummy values (e.g., random values)
    for (size_t i = 0; i < input_size; ++i) {
        input_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Example: random values between 0 and 1
    }

    // Optional: Check the input data shape for debugging
    std::cout << "Input data shape: (" << batch_size << ", " << input_size << ")" << std::endl;

    // Run inference
    try {
        std::vector<float> output = model_inference.run_inference(input_data);
        
        // Check output size for debugging
        std::cout << "Output size: " << output.size() << std::endl;

        // Render the output using OpenGL
        OpenGLRenderer renderer;
        renderer.render(output);
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return -1; // Exit or handle error appropriately
    }

    return 0;
}
