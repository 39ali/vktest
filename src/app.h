#pragma once

#include "camera.h"
#include "object3d.h"
#include "resource.h"

#include <memory>
#include <vector>

struct GLFWwindow;
class Renderer;

class App {
public:
  App(std::filesystem::path appDir);
  ~App();
  void run();

private:
  void initWindow();
  void initScene(const std::filesystem::path &appDir);

  Resources _resources;
  std::vector<Object3D> _objects;
  Camera _camera;
  GLFWwindow *_window = nullptr;
  std::unique_ptr<Renderer> _renderer;
};
