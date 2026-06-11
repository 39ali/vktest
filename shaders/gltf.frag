#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

vec3 lightDir = normalize(vec3(0.4, 0.8, 0.6));
void main() {

  float lighting = max(dot(normalize(inNormal), lightDir), 0.0) * 0.5 + 0.01;
  vec3 base = inColor.rgb;
  outColor = vec4(base * lighting, 1.0);
}
