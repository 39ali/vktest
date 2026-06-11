#pragma once

#include <glm/glm.hpp>

struct GLFWwindow;

struct CameraCullData {
  glm::mat4 view{1.0f};
  glm::vec4 frustum{1.0f};
  glm::vec2 zNearFar{0.01f, 1000.0f};
  glm::vec2 projectionScale{1.0f};
};

class Camera {
public:
  void update(GLFWwindow *window, float dt);
  void lookAt(const glm::vec3 &position, const glm::vec3 &target);
  glm::mat4 viewProjectionMatrix() const;
  CameraCullData cullData() const;

private:
  void onCursorPosition(double x, double y);

  glm::mat4 viewMatrix() const;
  glm::mat4 projectionMatrix() const;
  glm::vec3 forward() const;
  glm::vec3 right() const;

  glm::vec3 _position{0.0f, 0.0f, 6.0f};
  float _yaw = -90.0f;
  float _pitch = 0.0f;
  float _aspect = 16.0f / 9.0f;
  float _fovY = 45.0f;
  float _nearPlane = 0.01f;
  float _farPlane = 1000.0f;
  float _moveSpeed = 2.0f;
  float _mouseSensitivity = 0.12f;
  bool _mouseLook = false;
  bool _hasLastMousePosition = false;
  double _lastMouseX = 0.0;
  double _lastMouseY = 0.0;
};

