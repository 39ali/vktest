#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "resource.h"

#include <meshoptimizer.h>
#include <tiny_gltf.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/packing.hpp>

#include <iostream>

struct Vertex {
  glm::vec3 position{};
  glm::vec3 normal{0.0f, 0.0f, 1.0f};
};

struct CpuGeometry {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

uint32_t packCountsAndFlags(uint32_t vertexCount, uint32_t triangleCount,
                            uint32_t flags = 0) {
  return (vertexCount & 0xffu) | ((triangleCount & 0xffu) << 8u) |
         ((flags & 0xffffu) << 16u);
}

uint32_t packMeshletCone(const meshopt_Bounds &bounds) {
  return static_cast<uint8_t>(bounds.cone_axis_s8[0]) |
         (static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_axis_s8[1]))
          << 8u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_axis_s8[2]))
          << 16u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_cutoff_s8))
          << 24u);
}

uint32_t packTriangle(uint8_t a, uint8_t b, uint8_t c) {
  return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 10) |
         (static_cast<uint32_t>(c) << 20);
}

MeshletVertex packClusterVertex(const Vertex &vertex) {
  const glm::vec3 normal = glm::normalize(vertex.normal);

  MeshletVertex packed{};
  packed.positionXY = glm::packHalf2x16({vertex.position.x, vertex.position.y});
  packed.positionZNormalX = glm::packHalf2x16({vertex.position.z, normal.x});
  packed.normalYZ = glm::packHalf2x16({normal.y, normal.z});
  return packed;
}

template <typename T>
const T *accessorData(const tinygltf::Model &model,
                      const tinygltf::Accessor &accessor) {
  const auto &view = model.bufferViews[accessor.bufferView];
  const auto &buffer = model.buffers[view.buffer];
  return reinterpret_cast<const T *>(buffer.data.data() + view.byteOffset +
                                     accessor.byteOffset);
}

size_t accessorStride(const tinygltf::Model &model,
                      const tinygltf::Accessor &accessor) {
  const auto &view = model.bufferViews[accessor.bufferView];
  return accessor.ByteStride(view) > 0
             ? static_cast<size_t>(accessor.ByteStride(view))
             : tinygltf::GetComponentSizeInBytes(accessor.componentType) *
                   tinygltf::GetNumComponentsInType(accessor.type);
}

glm::mat4 nodeMatrix(const tinygltf::Node &node) {
  glm::mat4 matrix(1.0f);
  if (node.matrix.size() == 16) {
    matrix = glm::make_mat4x4(node.matrix.data());
  }
  if (node.translation.size() == 3) {
    matrix = glm::translate(matrix,
                            glm::vec3(node.translation[0], node.translation[1],
                                      node.translation[2]));
  }
  if (node.rotation.size() == 4) {
    const glm::quat q(static_cast<float>(node.rotation[3]),
                      static_cast<float>(node.rotation[0]),
                      static_cast<float>(node.rotation[1]),
                      static_cast<float>(node.rotation[2]));
    matrix *= glm::mat4_cast(q);
  }
  if (node.scale.size() == 3) {
    matrix = glm::scale(matrix,
                        glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
  }
  return matrix;
}

tinygltf::Model loadGltf(const std::filesystem::path &path) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string error;
  std::string warning;

  const std::string extension = path.extension().string();
  const bool loaded =
      extension == ".glb"
          ? loader.LoadBinaryFromFile(&model, &error, &warning, path.string())
          : loader.LoadASCIIFromFile(&model, &error, &warning, path.string());

  if (!warning.empty()) {
    std::cerr << "tinygltf warning: " << warning << '\n';
  }
  if (!loaded) {
    std::cerr << "tinygltf error: " << error << '\n';
  }
  assert(loaded && "failed to load glTF");

  return model;
}

void loadNode(const tinygltf::Model &model, const tinygltf::Node &node,
              const glm::mat4 &parentTransform, CpuGeometry &geometry) {
  const glm::mat4 transform = parentTransform * nodeMatrix(node);

  if (node.mesh >= 0) {
    const auto &mesh = model.meshes[node.mesh];
    for (const auto &primitive : mesh.primitives) {
      if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
        continue;
      }

      const auto positionIt = primitive.attributes.find("POSITION");
      if (positionIt == primitive.attributes.end()) {
        continue;
      }

      const auto &positionAccessor = model.accessors[positionIt->second];
      const auto *positions = accessorData<uint8_t>(model, positionAccessor);
      const size_t positionStride = accessorStride(model, positionAccessor);

      const uint8_t *normals = nullptr;
      size_t normalStride = 0;
      if (const auto it = primitive.attributes.find("NORMAL");
          it != primitive.attributes.end()) {
        const auto &accessor = model.accessors[it->second];
        normals = accessorData<uint8_t>(model, accessor);
        normalStride = accessorStride(model, accessor);
      }

      const uint32_t firstVertex =
          static_cast<uint32_t>(geometry.vertices.size());
      for (size_t i = 0; i < positionAccessor.count; ++i) {
        Vertex vertex{};
        const auto *p =
            reinterpret_cast<const float *>(positions + i * positionStride);
        vertex.position =
            glm::vec3(transform * glm::vec4(p[0], p[1], p[2], 1.0f));
        if (normals) {
          const auto *n =
              reinterpret_cast<const float *>(normals + i * normalStride);
          vertex.normal = glm::normalize(glm::mat3(transform) *
                                         glm::vec3(n[0], n[1], n[2]));
        }
        geometry.vertices.push_back(vertex);
      }

      if (primitive.indices >= 0) {
        const auto &accessor = model.accessors[primitive.indices];
        const auto *indexBytes = accessorData<uint8_t>(model, accessor);
        const size_t stride = accessorStride(model, accessor);
        for (size_t i = 0; i < accessor.count; ++i) {
          uint32_t index = 0;
          const auto *value = indexBytes + i * stride;
          switch (accessor.componentType) {
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            index = *reinterpret_cast<const uint8_t *>(value);
            break;
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            index = *reinterpret_cast<const uint16_t *>(value);
            break;
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            index = *reinterpret_cast<const uint32_t *>(value);
            break;
          default:
            assert(false && "unsupported glTF index component type");
            break;
          }
          geometry.indices.push_back(firstVertex + index);
        }
      } else {
        for (size_t i = 0; i < positionAccessor.count; ++i) {
          geometry.indices.push_back(firstVertex + static_cast<uint32_t>(i));
        }
      }
    }
  }

  for (const int childIndex : node.children) {
    loadNode(model, model.nodes[childIndex], transform, geometry);
  }
}

CpuGeometry flattenGeometry(const tinygltf::Model &model) {
  CpuGeometry geometry;
  const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
  for (const int nodeIndex : model.scenes[sceneIndex].nodes) {
    loadNode(model, model.nodes[nodeIndex], glm::mat4(1.0f), geometry);
  }

  assert(!geometry.vertices.empty() && !geometry.indices.empty() &&
         "glTF contains no triangle geometry");
  return geometry;
}

GltfModel buildMeshlets(const CpuGeometry &geometry) {
  constexpr size_t kMaxMeshletVertices = 64;
  constexpr size_t kMaxMeshletTriangles = 124;
  constexpr float kConeWeight = 0.0f;
  GltfModel gltfModel;
  gltfModel.packedClusterVertices.reserve(geometry.vertices.size());
  for (const Vertex &vertex : geometry.vertices) {
    gltfModel.packedClusterVertices.push_back(packClusterVertex(vertex));
  }

  const size_t maxMeshlets = meshopt_buildMeshletsBound(
      geometry.indices.size(), kMaxMeshletVertices, kMaxMeshletTriangles);

  std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
  std::vector<unsigned int> meshletVertices(maxMeshlets * kMaxMeshletVertices);
  std::vector<unsigned char> meshletTriangles(maxMeshlets *
                                              kMaxMeshletTriangles * 3);

  const size_t meshletCount = meshopt_buildMeshlets(
      meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
      geometry.indices.data(), geometry.indices.size(),
      reinterpret_cast<const float *>(geometry.vertices.data()),
      geometry.vertices.size(), sizeof(Vertex), kMaxMeshletVertices,
      kMaxMeshletTriangles, kConeWeight);

  gltfModel.packedMeshlets.reserve(meshletCount);
  for (size_t i = 0; i < meshletCount; ++i) {
    const meshopt_Meshlet &meshlet = meshlets[i];
    const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
        &meshletVertices[meshlet.vertex_offset],
        &meshletTriangles[meshlet.triangle_offset], meshlet.triangle_count,
        reinterpret_cast<const float *>(geometry.vertices.data()),
        geometry.vertices.size(), sizeof(Vertex));

    Meshlet meshletData{};
    meshletData.packedSphere0 =
        glm::packHalf2x16({bounds.center[0], bounds.center[1]});
    meshletData.packedSphere1 =
        glm::packHalf2x16({bounds.center[2], bounds.radius});
    meshletData.vertexOffset =
        static_cast<uint32_t>(gltfModel.packedMeshletVertexRefs.size());
    meshletData.triangleOffset =
        static_cast<uint32_t>(gltfModel.packedClusterTriangles.size());
    meshletData.packedCountsAndFlags =
        packCountsAndFlags(static_cast<uint32_t>(meshlet.vertex_count),
                           static_cast<uint32_t>(meshlet.triangle_count));
    meshletData.boundingCone = packMeshletCone(bounds);

    for (uint32_t vertex = 0; vertex < meshlet.vertex_count; ++vertex) {
      const uint32_t vertexIndex =
          meshletVertices[meshlet.vertex_offset + vertex];
      gltfModel.packedMeshletVertexRefs.push_back(vertexIndex);
    }
    for (uint32_t triangle = 0; triangle < meshlet.triangle_count; ++triangle) {
      const size_t triangleOffset = meshlet.triangle_offset + triangle * 3;
      gltfModel.packedClusterTriangles.push_back(
          packTriangle(meshletTriangles[triangleOffset],
                       meshletTriangles[triangleOffset + 1],
                       meshletTriangles[triangleOffset + 2]));
    }
    gltfModel.packedMeshlets.push_back(meshletData);
  }

  return gltfModel;
}

Resources::Resources(std::filesystem::path baseDir)
    : _baseDir(std::move(baseDir)) {}

std::vector<char>
Resources::readBinary(const std::filesystem::path &path) const {
  const auto resolvedPath = _baseDir / path;
  std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
  if (!file) {
    std::cerr << "failed to open file: " << resolvedPath << '\n';
    assert(false && "failed to open file");
    return {};
  }

  const auto size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return buffer;
}

GltfModel Resources::loadGltfModel(const std::filesystem::path &path) const {
  const auto resolvedPath = _baseDir / path;
  return buildMeshlets(flattenGeometry(loadGltf(resolvedPath)));
}
