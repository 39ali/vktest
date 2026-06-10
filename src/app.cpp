#include "app.h"

#include "renderer.h"

#include <GLFW/glfw3.h>

#include <glm/gtc/quaternion.hpp>

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

constexpr uint32_t kWidth = 1920;
constexpr uint32_t kHeight = 1080;

App::App(std::filesystem::path appDir) : _resources(appDir) {

  initWindow();
  _renderer = std::make_unique<Renderer>(_window, appDir);
  _renderer->init();
  initScene(appDir);
}

App::~App() {
  _renderer.reset();
  if (_window) {
    glfwDestroyWindow(_window);
    _window = nullptr;
  }
  glfwTerminate();
}

void App::initWindow() {
  const int glfwInitialized = glfwInit();
  assert(glfwInitialized && "failed to initialize GLFW");

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  _window =
      glfwCreateWindow(kWidth, kHeight, "vktest glTF Vulkan", nullptr, nullptr);
  assert(_window && "failed to create GLFW window");
}

void App::initScene(const std::filesystem::path &appDir) {
  const std::filesystem::path modelPath = "assets/Duck.glb";
  GltfModel model = _resources.loadGltfModel(modelPath);
  std::cout << "meshlets: " << model.packedMeshlets.size()
            << ", triangles: " << model.packedClusterTriangles.size() << '\n';
  const MeshId duckMesh = _renderer->loadModel(model);

  constexpr int kObjectCount = 10'000; // 100'000;
  constexpr float kSceneRadius = 4.0f;
  _objects.reserve(kObjectCount);

  std::random_device randomDevice;
  std::mt19937 generator(randomDevice());
  std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);

  for (int i = 0; i < kObjectCount; ++i) {
    glm::vec3 position{};
    do {
      position = glm::vec3{randomFloat(generator) * 2.0f - 1.0f,
                           randomFloat(generator) * 2.0f - 1.0f,
                           randomFloat(generator) * 2.0f - 1.0f} *
                 kSceneRadius;
    } while (glm::dot(position, position) > kSceneRadius * kSceneRadius);

    Object3D object(duckMesh);
    object.position = position;
    const float scale = 0.08f + randomFloat(generator) * 0.16f;
    object.scale = glm::vec3{scale};
    const float yaw = randomFloat(generator) * glm::two_pi<float>();
    const float pitch = (randomFloat(generator) - 0.5f) * glm::radians(25.0f);
    const float roll = (randomFloat(generator) - 0.5f) * glm::radians(25.0f);
    object.rotation =
        glm::angleAxis(yaw + glm::radians(90.0f), glm::vec3{0.0f, 1.0f, 0.0f}) *
        glm::angleAxis(pitch, glm::vec3{1.0f, 0.0f, 0.0f}) *
        glm::angleAxis(roll, glm::vec3{0.0f, 0.0f, 1.0f});
    object.color = {randomFloat(generator), randomFloat(generator),
                    randomFloat(generator), 1.0f};
    _objects.push_back(object);
  }

  _camera.lookAt(glm::vec3{0.0f, 0.0f, kSceneRadius * 2.4f}, glm::vec3{0.0f});
  _renderer->setObjects(_objects);
}

void App::run() {
  auto previousTime = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(_window)) {
    const auto now = std::chrono::steady_clock::now();
    const float dt =
        std::chrono::duration<float, std::milli>(now - previousTime).count();
    previousTime = now;

    _camera.update(_window, dt);
    _renderer->render(_camera.viewProjectionMatrix(), _camera.cullData(), dt);
    glfwPollEvents();
  }
  _renderer->waitIdle();
}
