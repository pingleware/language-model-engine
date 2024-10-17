// opengl_renderer.h
#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>

class OpenGLRenderer {
public:
    OpenGLRenderer();
    ~OpenGLRenderer();
    void render(const std::vector<float>& model_output);
    
private:
    GLFWwindow* window;
    void init_opengl();
};