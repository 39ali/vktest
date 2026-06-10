#pragma once

#include "shared_structs.h"

#include <cstdint>
#include <filesystem>
#include <vector>

using MeshId = uint32_t;

struct GltfModel {
  std::vector<MeshletVertex> packedClusterVertices;
  std::vector<uint32_t> packedMeshletVertexRefs;
  std::vector<uint32_t> packedClusterTriangles;
  std::vector<Meshlet> packedMeshlets;
};

class Resources {
public:
  Resources(std::filesystem::path baseDir);

  std::vector<char> readBinary(const std::filesystem::path &path) const;
  GltfModel loadGltfModel(const std::filesystem::path &path) const;

private:
  std::filesystem::path _baseDir;
};
