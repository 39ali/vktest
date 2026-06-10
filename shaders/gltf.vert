#version 450
#extension GL_GOOGLE_include_directive : require

#include "shared_structs.h"

layout(push_constant) uniform PushConstants {
  mat4 mvp;
}
pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;

layout(set = 0, binding = 0, std430) readonly buffer InstanceBuffer {
  InstanceData instances[];
};

layout(set = 0, binding = 1, std430) readonly buffer MeshletBuffer {
  Meshlet meshlets[];
};

layout(set = 0, binding = 2, std430) readonly buffer VertexBuffer {
  MeshletVertex vertices[];
};

layout(set = 0, binding = 3, std430) readonly buffer ClusterTriangleBuffer {
  uint clusterTriangles[];
};

layout(set = 0, binding = 5, std430) readonly buffer VisibleMeshletBuffer {
  VisibleMeshlet visibleMeshlets[];
};

layout(set = 0, binding = 9, std430) readonly buffer MeshletVertexRefBuffer {
  uint meshletVertexRefs[];
};

vec4 unpackColor(uint color) {
  return vec4(float(color & 0xffu), float((color >> 8) & 0xffu),
              float((color >> 16) & 0xffu), float((color >> 24) & 0xffu)) /
         255.0;
}

void main() {
  uint localVertexIndex = gl_VertexIndex;
  VisibleMeshlet visibleMeshlet = visibleMeshlets[gl_InstanceIndex];
  InstanceData instance = instances[visibleMeshlet.instanceId];
  Meshlet meshlet = meshlets[visibleMeshlet.meshletId];

  uint triangleIndex = localVertexIndex / 3u;
  uint triangleCorner = localVertexIndex % 3u;
  if (triangleIndex >= meshletTriangleCount(meshlet)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    outNormal = vec3(0.0, 0.0, 1.0);
    outColor = vec4(0.0);
    return;
  }

  uint packedTriangle = clusterTriangles[meshlet.triangleOffset + triangleIndex];
  uint clusterVertexIndex = (packedTriangle >> (triangleCorner * 10u)) & 0x3ffu;
  uint vertexIndex = meshletVertexRefs[meshlet.vertexOffset + clusterVertexIndex];
  MeshletVertex vertex = vertices[vertexIndex];

  vec2 positionXY = unpackHalf2x16(vertex.positionXY);
  vec2 positionZNormalX = unpackHalf2x16(vertex.positionZNormalX);
  vec2 normalYZ = unpackHalf2x16(vertex.normalYZ);
  vec3 localPosition =
      vec3(positionXY.x, positionXY.y, positionZNormalX.x);
  vec3 position = localPosition;
  vec3 normal = normalize(vec3(positionZNormalX.y, normalYZ.x, normalYZ.y));

  vec3 translation = instance.translationScaleX.xyz;
  vec3 scale = vec3(instance.translationScaleX.w, instance.scaleYZ.x,
                    instance.scaleYZ.y);
  vec3 worldPosition =
      translation + rotateByQuat(position * scale, instance.rotation);

  gl_Position = pc.mvp * vec4(worldPosition, 1.0);
  outNormal = normalize(rotateByQuat(normal, instance.rotation));
  outColor = unpackColor(instance.packedColor);
}
