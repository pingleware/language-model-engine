cmake_minimum_required(VERSION 3.10)
project(LanguageModelEngine)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the path to the GLFW installation
if (APPLE)
    set(GLEW_INCLUDE_DIRS "/opt/homebrew/Cellar/glew/2.2.0_1/include")
    set(GLEW_LIBRARY "/opt/homebrew/Cellar/glew/2.2.0_1/lib/libGLEW.dylib")
    set(GLFW_INCLUDE_DIRS "/opt/homebrew/Cellar/glfw/3.4/include")
    set(GLFW_LIBRARIES "/opt/homebrew/Cellar/glfw/3.4/lib/libglfw.dylib")
    set(ONNXRUNTIME_ROOTDIR "/opt/homebrew/Cellar/onnxruntime/1.17.1")
    set(ONNXRUNTIME_INCLUDE_DIRS "/opt/homebrew/Cellar/onnxruntime/1.17.1/include")
    set(ONNXRUNTIME_LIBRARY "/opt/homebrew/Cellar/onnxruntime/1.17.1/lib/libonnxruntime.dylib")
endif()

# Find OpenGL, GLEW, and GLFW
find_package(onnxruntime REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS} include)

# Source files
add_executable(LanguageModelEngine 
    src/main.cpp 
    src/model_inference.cpp 
    src/opengl_renderer.cpp
)

# Link OpenGL, GLEW, and GLFW
target_link_libraries(LanguageModelEngine ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLFW_LIBRARIES} ${ONNXRUNTIME_LIBRARY})