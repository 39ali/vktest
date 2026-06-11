#pragma once

#include "camera.h"
#include "object3d.h"
#include "resource.h"
#include "vk_context.h"

#include <filesystem>
#include <vector>
#include <vulkan/vulkan.h>

struct GLFWwindow;

template <typename T> struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct MeshUploadInfo {
  uint32_t firstMeshlet = 0;
  uint32_t meshletCount = 0;
};

struct RenderCounters {
  uint32_t visibleMeshletCount = 0;
  uint32_t visibleTriangleCount = 0;
};

struct RenderBucket {
  Buffer<MeshletInstance> candidateMeshletBuffer;
  Buffer<MeshletInstance> visibleMeshletBuffer;
  Buffer<DrawIndirectCommand> drawArgumentBuffer;
  Buffer<RenderCounters> counterBuffer;
  Buffer<MeshletDrawMeta> meshletDrawMetaBuffer;
  Buffer<RenderCounters> counterReadbackBuffers[VkContext::kMaxFramesInFlight];
};

enum class IndirectMode { SingleDraw, MultiDraw };

struct RenderStats {
  uint32_t visibleMeshlets = 0;
  uint32_t totalMeshlets = 0;
  uint32_t visibleTriangles = 0;
  uint64_t totalTriangles = 0;
};

class Renderer {
public:
  Renderer(GLFWwindow *window, std::filesystem::path appDir);
  ~Renderer();

  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  void init();
  MeshId loadModel(const GltfModel &model);
  void setObjects(const std::vector<Object3D> &objects);
  void render(const glm::mat4 &viewProjection, const CameraCullData &cullData,
              float dt);
  void waitIdle();

private:
  void setIndirectMode(IndirectMode mode);
  void cleanup();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createCullingPipeline();
  void createImgui();
  void createDescriptorPool();
  void createDescriptorSet();
  void updateInstanceDescriptorSet();
  void updateMeshDescriptorSet();
  void updateRenderBucketDescriptorSet();
  void rebuildMeshBuffers();
  void uploadRenderBucket();
  void ensureCounterReadbackBuffer(size_t frameIndex);
  void readCompletedCounters(size_t frameIndex);
  template <typename T>
  bool uploadHostBuffer(const std::vector<T> &source, VkBufferUsageFlags usage,
                        Buffer<T> &target, size_t &capacity);
  template <typename T>
  void uploadDeviceLocalBuffer(const std::vector<T> &source,
                               VkBufferUsageFlags usage, Buffer<T> &target);
  template <typename T>
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, Buffer<T> &buffer);
  template <typename T> void destroyBuffer(Buffer<T> &buffer);
  void recordCommandBuffer(const FrameContext &frame,
                           const glm::mat4 &viewProjection,
                           const CameraCullData &cullData, float dt);
  void recordComputeCull(VkCommandBuffer commandBuffer,
                         VkBuffer counterReadbackBuffer,
                         const CameraCullData &cullData);
  void renderImgui(VkCommandBuffer commandBuffer, float dt);
  void cleanupImgui();

  VkContext _vkContext;
  Resources _resources;
  std::vector<Meshlet> _meshlets;
  std::vector<MeshletVertex> _clusterVertices;
  std::vector<uint32_t> _meshletVertexRefs;
  std::vector<uint32_t> _packedClusterTriangles;
  std::vector<MeshUploadInfo> _meshes;
  std::vector<InstanceData> _frameInstances;
  std::vector<MeshletInstance> _candidateMeshlets;
  std::vector<MeshletDrawMeta> _meshletDrawMetas;
  uint32_t _indirectDrawCount = 0;

  IndirectMode _indirectMode = IndirectMode::MultiDraw;
  VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
  VkPipeline _graphicsPipeline = VK_NULL_HANDLE;
  VkPipeline _computePipeline = VK_NULL_HANDLE;
  VkDescriptorPool _descriptorPool = VK_NULL_HANDLE;
  VkDescriptorPool _imguiDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet _descriptorSet = VK_NULL_HANDLE;
  Buffer<Meshlet> _meshletBuffer;
  Buffer<MeshletVertex> _clusterVertexBuffer;
  Buffer<uint32_t> _meshletVertexRefBuffer;
  Buffer<uint32_t> _clusterTriangleBuffer;
  Buffer<InstanceData> _instanceBuffer;
  RenderBucket _renderBucket;
  size_t _instanceBufferCapacity = 0;
  size_t _candidateMeshletCapacity = 0;
  size_t _visibleMeshletCapacity = 0;
  size_t _drawArgumentCapacity = 0;
  size_t _meshletDrawMetaCapacity = 0;
  RenderStats _stats;
  bool _switchIndirectModeRequested = false;
};
