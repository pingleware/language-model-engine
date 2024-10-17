// opengl_renderer.cpp
#include "opengl_renderer.h"
#include <iostream>

OpenGLRenderer::OpenGLRenderer() {
    init_opengl();
}

OpenGLRenderer::~OpenGLRenderer() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void OpenGLRenderer::init_opengl() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(-1);
    }

    window = glfwCreateWindow(800, 600, "Model Output", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);
    // After creating an OpenGL context (e.g., after creating a window with GLFW)
    #ifndef __APPLE__
    glewExperimental = GL_TRUE; // Make sure that GLEW uses the modern OpenGL
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return;
    }
    #endif
}

void OpenGLRenderer::render(const std::vector<float>& model_output) {
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render model output (for simplicity, let's assume a line graph)
        glBegin(GL_LINES);
        for (size_t i = 0; i < model_output.size() - 1; ++i) {
            glVertex2f(i / float(model_output.size()), model_output[i]);
            glVertex2f((i + 1) / float(model_output.size()), model_output[i + 1]);
        }
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}