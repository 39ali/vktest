#define NOMINMAX
#include "renderer.h"

#include "math_helper.h"
#include "object3d.h"
#include "resource.h"

#include <GLFW/glfw3.h>
#include <array>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace {

static_assert(sizeof(DrawIndirectCommand) == sizeof(VkDrawIndirectCommand));

constexpr VkShaderStageFlags kPushConstantStages =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

struct PushConstants {
  glm::mat4 viewProjection{1.0f};
};

struct ComputePushConstants {
  glm::mat4 view{1.0f};
  glm::vec4 frustum{1.0f};
  glm::vec2 zNearFar{0.01f, 1000.0f};
  uint32_t candidateCount = 0;
  uint32_t mode = 0;
  uint32_t meshletDrawCount = 0;
  uint32_t padding0 = 0;
};

struct GpuCounters {
  uint32_t visibleMeshletCount = 0;
  uint32_t visibleTriangleCount = 0;
};

static_assert(sizeof(GpuCounters) == 8);

uint32_t packColor(const glm::vec4 &color) {
  const auto packChannel = [](float value) {
    return static_cast<uint32_t>(clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
  };
  return packChannel(color.r) | (packChannel(color.g) << 8) |
         (packChannel(color.b) << 16) | (packChannel(color.a) << 24);
}

InstanceData makeInstanceData(const Object3D &object) {
  InstanceData instance{};
  instance.translationScaleX = {object.position.x, object.position.y,
                                object.position.z, object.scale.x};
  instance.scaleYZ = {object.scale.y, object.scale.z};
  instance.rotation = {object.rotation.x, object.rotation.y, object.rotation.z,
                       object.rotation.w};
  instance.packedColor = packColor(object.color);
  return instance;
}

} // namespace

Renderer::Renderer(GLFWwindow *window, std::filesystem::path appDir)
    : _vkContext(window), _resources(std::move(appDir)), _window(window) {}

Renderer::~Renderer() { cleanup(); }

void Renderer::setIndirectMode(IndirectMode mode) {
  if (mode == IndirectMode::MultiDraw &&
      _vkContext.supportsMultiDrawIndirect()) {
    _indirectMode = IndirectMode::MultiDraw;
  } else {
    _indirectMode = IndirectMode::SingleDraw;
  }

  _indirectDrawCount = _indirectMode == IndirectMode::MultiDraw
                           ? static_cast<uint32_t>(_meshletDrawMetas.size())
                           : (_candidateMeshlets.empty() ? 0u : 1u);
  if (!_candidateMeshlets.empty()) {
    uploadRenderBucket();
  }
}

void Renderer::init() {
  _vkContext.init();
  setIndirectMode(IndirectMode::MultiDraw);
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCullingPipeline();
  createFrameResources();
  createImgui();
}

void Renderer::createFrameResources() {
  createDescriptorPool();
  createDescriptorSet();
}

void Renderer::waitIdle() { _vkContext.waitIdle(); }

void Renderer::cleanup() {
  if (_vkContext.device() != VK_NULL_HANDLE) {
    cleanupImgui();
    for (Buffer &buffer : _opaqueBucket.counterReadbackBuffers) {
      destroyBuffer(buffer);
    }
    destroyBuffer(_opaqueBucket.meshletDrawMetaBuffer);
    destroyBuffer(_opaqueBucket.drawArgumentBuffer);
    destroyBuffer(_opaqueBucket.counterBuffer);
    destroyBuffer(_opaqueBucket.visibleMeshletBuffer);
    destroyBuffer(_opaqueBucket.candidateMeshletBuffer);
    destroyBuffer(_instanceBuffer);
    destroyBuffer(_clusterTriangleBuffer);
    destroyBuffer(_meshletVertexRefBuffer);
    destroyBuffer(_clusterVertexBuffer);
    destroyBuffer(_meshletBuffer);
    vkDestroyPipeline(_vkContext.device(), _computePipeline, nullptr);
    vkDestroyPipeline(_vkContext.device(), _graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(_vkContext.device(), _pipelineLayout, nullptr);
    vkDestroyDescriptorPool(_vkContext.device(), _descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(_vkContext.device(), _descriptorSetLayout,
                                 nullptr);
    _vkContext.cleanup();
  }
}

void Renderer::createDescriptorSetLayout() {
  std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
  for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
    bindings[binding].binding = binding;
    bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[binding].descriptorCount = 1;
    bindings[binding].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  const VkResult createDescriptorSetLayoutResult = vkCreateDescriptorSetLayout(
      _vkContext.device(), &layoutInfo, nullptr, &_descriptorSetLayout);
  assert(createDescriptorSetLayoutResult == VK_SUCCESS &&
         "failed to create descriptor set layout");
}

void Renderer::createGraphicsPipeline() {
  const auto vertShaderCode = _resources.readBinary("shaders/gltf.vert.spv");
  const auto fragShaderCode = _resources.readBinary("shaders/gltf.frag.spv");
  const VkShaderModule vertShaderModule =
      _vkContext.createShaderModule(vertShaderCode);
  const VkShaderModule fragShaderModule =
      _vkContext.createShaderModule(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";
  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkViewport viewport{};
  viewport.width = static_cast<float>(renderTarget.extent.width);
  viewport.height = static_cast<float>(renderTarget.extent.height);
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{{0, 0}, renderTarget.extent};
  VkPipelineViewportStateCreateInfo viewportState{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

  VkPipelineMultisampleStateCreateInfo multisampling{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo colorBlending{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = kPushConstantStages;
  pushConstantRange.offset = 0;
  pushConstantRange.size =
      maxValue(sizeof(PushConstants), sizeof(ComputePushConstants));

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &_descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
  const VkResult createPipelineLayoutResult = vkCreatePipelineLayout(
      _vkContext.device(), &pipelineLayoutInfo, nullptr, &_pipelineLayout);
  assert(createPipelineLayoutResult == VK_SUCCESS &&
         "failed to create pipeline layout");

  VkGraphicsPipelineCreateInfo pipelineInfo{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  VkPipelineRenderingCreateInfo renderingInfo{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachmentFormats = &renderTarget.colorFormat;
  renderingInfo.depthAttachmentFormat = renderTarget.depthFormat;

  pipelineInfo.pNext = &renderingInfo;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = _pipelineLayout;
  pipelineInfo.renderPass = VK_NULL_HANDLE;
  pipelineInfo.subpass = 0;

  const VkResult createGraphicsPipelineResult =
      vkCreateGraphicsPipelines(_vkContext.device(), VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr, &_graphicsPipeline);
  assert(createGraphicsPipelineResult == VK_SUCCESS &&
         "failed to create graphics pipeline");

  vkDestroyShaderModule(_vkContext.device(), fragShaderModule, nullptr);
  vkDestroyShaderModule(_vkContext.device(), vertShaderModule, nullptr);
}

void Renderer::createCullingPipeline() {
  const auto shaderCode =
      _resources.readBinary("shaders/meshlet_cull.comp.spv");
  const VkShaderModule shaderModule = _vkContext.createShaderModule(shaderCode);

  VkPipelineShaderStageCreateInfo shaderStageInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName = "main";

  VkComputePipelineCreateInfo pipelineInfo{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipelineInfo.stage = shaderStageInfo;
  pipelineInfo.layout = _pipelineLayout;

  const VkResult createComputePipelineResult =
      vkCreateComputePipelines(_vkContext.device(), VK_NULL_HANDLE, 1,
                               &pipelineInfo, nullptr, &_computePipeline);
  assert(createComputePipelineResult == VK_SUCCESS &&
         "failed to create compute pipeline");

  vkDestroyShaderModule(_vkContext.device(), shaderModule, nullptr);
}

void Renderer::createImgui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();

  std::array<VkDescriptorPoolSize, 1> poolSizes{{
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
  }};

  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolInfo.maxSets = 16;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  const VkResult createPoolResult = vkCreateDescriptorPool(
      _vkContext.device(), &poolInfo, nullptr, &_imguiDescriptorPool);
  assert(createPoolResult == VK_SUCCESS &&
         "failed to create ImGui descriptor pool");

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkPipelineRenderingCreateInfo renderingInfo{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachmentFormats = &renderTarget.colorFormat;
  renderingInfo.depthAttachmentFormat = renderTarget.depthFormat;

  ImGui_ImplGlfw_InitForVulkan(_window, true);
  ImGui_ImplVulkan_InitInfo initInfo{};
  initInfo.Instance = _vkContext.instance();
  initInfo.PhysicalDevice = _vkContext.physicalDevice();
  initInfo.Device = _vkContext.device();
  initInfo.QueueFamily = _vkContext.graphicsQueueFamily();
  initInfo.Queue = _vkContext.graphicsQueue();
  initInfo.DescriptorPool = _imguiDescriptorPool;
  initInfo.MinImageCount = _vkContext.swapChainImageCount();
  initInfo.ImageCount = _vkContext.swapChainImageCount();
  initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  initInfo.UseDynamicRendering = true;
  initInfo.PipelineRenderingCreateInfo = renderingInfo;
  ImGui_ImplVulkan_Init(&initInfo);
}

void Renderer::createDescriptorPool() {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 10;

  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 1;

  const VkResult createDescriptorPoolResult = vkCreateDescriptorPool(
      _vkContext.device(), &poolInfo, nullptr, &_descriptorPool);
  assert(createDescriptorPoolResult == VK_SUCCESS &&
         "failed to create descriptor pool");
}

void Renderer::createDescriptorSet() {
  VkDescriptorSetLayout layout = _descriptorSetLayout;
  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = _descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;

  const VkResult allocateDescriptorSetResult = vkAllocateDescriptorSets(
      _vkContext.device(), &allocInfo, &_descriptorSet);
  assert(allocateDescriptorSetResult == VK_SUCCESS &&
         "failed to allocate descriptor set");
}

void Renderer::updateInstanceDescriptorSet() {
  const VkDeviceSize bufferSize =
      sizeof(InstanceData) * _instanceBufferCapacity;

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = _instanceBuffer.buffer;
  bufferInfo.offset = 0;
  bufferInfo.range = bufferSize;

  VkWriteDescriptorSet descriptorWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  descriptorWrite.dstSet = _descriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorWrite.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(_vkContext.device(), 1, &descriptorWrite, 0, nullptr);
}

void Renderer::updateMeshDescriptorSet() {
  if (_meshlets.empty() || _clusterVertices.empty() ||
      _meshletVertexRefs.empty() || _packedClusterTriangles.empty()) {
    return;
  }

  std::array<VkDescriptorBufferInfo, 4> bufferInfos{};
  bufferInfos[0].buffer = _meshletBuffer.buffer;
  bufferInfos[0].range = sizeof(Meshlet) * _meshlets.size();
  bufferInfos[1].buffer = _clusterVertexBuffer.buffer;
  bufferInfos[1].range = sizeof(MeshletVertex) * _clusterVertices.size();
  bufferInfos[2].buffer = _clusterTriangleBuffer.buffer;
  bufferInfos[2].range = sizeof(uint32_t) * _packedClusterTriangles.size();
  bufferInfos[3].buffer = _meshletVertexRefBuffer.buffer;
  bufferInfos[3].range = sizeof(uint32_t) * _meshletVertexRefs.size();

  constexpr std::array<uint32_t, 4> kBindings = {1, 2, 3, 9};
  std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
  for (uint32_t writeIndex = 0; writeIndex < descriptorWrites.size();
       ++writeIndex) {
    VkWriteDescriptorSet &descriptorWrite = descriptorWrites[writeIndex];
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = _descriptorSet;
    descriptorWrite.dstBinding = kBindings[writeIndex];
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.pBufferInfo = &bufferInfos[writeIndex];
  }

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

void Renderer::updateRenderBucketDescriptorSet() {
  if (_candidateMeshletCapacity == 0 || _visibleMeshletCapacity == 0 ||
      _drawArgumentCapacity == 0 || _meshletDrawMetaCapacity == 0 ||
      _opaqueBucket.counterBuffer.buffer == VK_NULL_HANDLE) {
    return;
  }

  std::array<VkDescriptorBufferInfo, 5> bufferInfos{};
  bufferInfos[0].buffer = _opaqueBucket.candidateMeshletBuffer.buffer;
  bufferInfos[0].range = sizeof(CandidateMeshlet) * _candidateMeshletCapacity;
  bufferInfos[1].buffer = _opaqueBucket.visibleMeshletBuffer.buffer;
  bufferInfos[1].range = sizeof(VisibleMeshlet) * _visibleMeshletCapacity;
  bufferInfos[2].buffer = _opaqueBucket.drawArgumentBuffer.buffer;
  bufferInfos[2].range = sizeof(DrawIndirectCommand) * _drawArgumentCapacity;
  bufferInfos[3].buffer = _opaqueBucket.counterBuffer.buffer;
  bufferInfos[3].range = sizeof(GpuCounters);
  bufferInfos[4].buffer = _opaqueBucket.meshletDrawMetaBuffer.buffer;
  bufferInfos[4].range = sizeof(MeshletDrawMeta) * _meshletDrawMetaCapacity;

  std::array<VkWriteDescriptorSet, 5> descriptorWrites{};
  for (uint32_t writeIndex = 0; writeIndex < descriptorWrites.size();
       ++writeIndex) {
    VkWriteDescriptorSet &descriptorWrite = descriptorWrites[writeIndex];
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = _descriptorSet;
    descriptorWrite.dstBinding = 4 + writeIndex;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.pBufferInfo = &bufferInfos[writeIndex];
  }

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

MeshId Renderer::loadModel(const GltfModel &model) {
  MeshUploadInfo info{};
  info.firstMeshlet = static_cast<uint32_t>(_meshlets.size());
  info.meshletCount = static_cast<uint32_t>(model.packedMeshlets.size());
  const uint32_t clusterVertexBase =
      static_cast<uint32_t>(_clusterVertices.size());
  const uint32_t meshletVertexRefBase =
      static_cast<uint32_t>(_meshletVertexRefs.size());
  const uint32_t clusterTriangleBase =
      static_cast<uint32_t>(_packedClusterTriangles.size());

  _packedClusterTriangles.insert(_packedClusterTriangles.end(),
                                 model.packedClusterTriangles.begin(),
                                 model.packedClusterTriangles.end());

  _clusterVertices.insert(_clusterVertices.end(),
                          model.packedClusterVertices.begin(),
                          model.packedClusterVertices.end());

  _meshletVertexRefs.reserve(_meshletVertexRefs.size() +
                             model.packedMeshletVertexRefs.size());
  for (uint32_t vertexRef : model.packedMeshletVertexRefs) {
    _meshletVertexRefs.push_back(clusterVertexBase + vertexRef);
  }

  _meshlets.insert(_meshlets.end(), model.packedMeshlets.begin(),
                   model.packedMeshlets.end());
  for (auto it = _meshlets.begin() + info.firstMeshlet; it != _meshlets.end();
       ++it) {
    Meshlet &meshlet = *it;
    meshlet.vertexOffset += meshletVertexRefBase;
    meshlet.triangleOffset += clusterTriangleBase;
  }
  _meshes.push_back(info);

  rebuildMeshBuffers();
  return static_cast<MeshId>(_meshes.size() - 1);
}

void Renderer::rebuildMeshBuffers() {

  uploadDeviceLocalBuffer(_meshlets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletBuffer);
  uploadDeviceLocalBuffer(_clusterVertices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _clusterVertexBuffer);
  uploadDeviceLocalBuffer(_meshletVertexRefs,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _meshletVertexRefBuffer);
  uploadDeviceLocalBuffer(_packedClusterTriangles,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          _clusterTriangleBuffer);
  updateMeshDescriptorSet();
}

void Renderer::uploadObjects(const std::vector<Object3D> &objects) {
  _frameInstances.clear();
  _candidateMeshlets.clear();
  _meshletDrawMetas.clear();
  _totalCandidateTriangles = 0;
  _frameInstances.reserve(objects.size());
  std::vector<uint32_t> meshletCandidateCounts(_meshlets.size(), 0);

  for (const Object3D &object : objects) {
    if (object.meshId >= _meshes.size()) {
      continue;
    }

    const uint32_t instanceId = static_cast<uint32_t>(_frameInstances.size());
    _frameInstances.push_back(makeInstanceData(object));

    const MeshUploadInfo &mesh = _meshes[object.meshId];
    for (uint32_t meshletOffset = 0; meshletOffset < mesh.meshletCount;
         ++meshletOffset) {
      const uint32_t meshletId = mesh.firstMeshlet + meshletOffset;
      const uint32_t triangleCount = meshletTriangleCount(_meshlets[meshletId]);
      if (triangleCount == 0) {
        continue;
      }

      _candidateMeshlets.push_back({instanceId, meshletId});
      ++meshletCandidateCounts[meshletId];
      _totalCandidateTriangles += triangleCount;
    }
  }

  _meshletDrawMetas.resize(_meshlets.size());
  uint32_t visibleInstanceOffset = 0;
  for (uint32_t meshletId = 0; meshletId < _meshletDrawMetas.size();
       ++meshletId) {
    _meshletDrawMetas[meshletId] = {visibleInstanceOffset, meshletId};
    visibleInstanceOffset += meshletCandidateCounts[meshletId];
  }

  _indirectDrawCount = _indirectMode == IndirectMode::MultiDraw
                           ? static_cast<uint32_t>(_meshletDrawMetas.size())
                           : (_candidateMeshlets.empty() ? 0u : 1u);
  _stats.totalMeshlets = static_cast<uint32_t>(_candidateMeshlets.size());
  _stats.totalTriangles = _totalCandidateTriangles;

  if (_frameInstances.empty()) {
    return;
  }

  if (_frameInstances.size() > _instanceBufferCapacity) {
    vkDeviceWaitIdle(_vkContext.device());
    destroyBuffer(_instanceBuffer);
    _instanceBufferCapacity = _frameInstances.size();
    const VkDeviceSize capacitySize =
        sizeof(InstanceData) * _instanceBufferCapacity;
    createBuffer(capacitySize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 _instanceBuffer);
    updateInstanceDescriptorSet();
  }

  const VkDeviceSize uploadSize =
      sizeof(_frameInstances[0]) * _frameInstances.size();

  void *data = nullptr;
  vkMapMemory(_vkContext.device(), _instanceBuffer.memory, 0, uploadSize, 0,
              &data);
  std::memcpy(data, _frameInstances.data(), static_cast<size_t>(uploadSize));
  vkUnmapMemory(_vkContext.device(), _instanceBuffer.memory);

  uploadRenderBucket();
}

void Renderer::setObjects(const std::vector<Object3D> &objects) {
  uploadObjects(objects);
}

template <typename T>
bool Renderer::uploadHostBuffer(const std::vector<T> &source,
                                VkBufferUsageFlags usage, Buffer &target,
                                size_t &capacity) {
  if (source.empty()) {
    return false;
  }

  bool recreated = false;
  const VkDeviceSize bufferSize = sizeof(source[0]) * source.size();
  if (source.size() > capacity) {
    vkDeviceWaitIdle(_vkContext.device());
    destroyBuffer(target);
    capacity = source.size();
    createBuffer(sizeof(source[0]) * capacity, usage,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 target);
    recreated = true;
  }

  void *data = nullptr;
  vkMapMemory(_vkContext.device(), target.memory, 0, bufferSize, 0, &data);
  std::memcpy(data, source.data(), static_cast<size_t>(bufferSize));
  vkUnmapMemory(_vkContext.device(), target.memory);
  return recreated;
}

void Renderer::uploadRenderBucket() {
  bool recreated = uploadHostBuffer(
      _candidateMeshlets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      _opaqueBucket.candidateMeshletBuffer, _candidateMeshletCapacity);

  recreated |= uploadHostBuffer(
      _meshletDrawMetas, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      _opaqueBucket.meshletDrawMetaBuffer, _meshletDrawMetaCapacity);

  if (_candidateMeshlets.size() > _visibleMeshletCapacity) {
    vkDeviceWaitIdle(_vkContext.device());
    destroyBuffer(_opaqueBucket.visibleMeshletBuffer);
    _visibleMeshletCapacity = _candidateMeshlets.size();
    createBuffer(sizeof(VisibleMeshlet) * _visibleMeshletCapacity,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 _opaqueBucket.visibleMeshletBuffer);
    recreated = true;
  }

  const size_t requiredDrawArgumentCount =
      _indirectMode == IndirectMode::MultiDraw
          ? maxValue<size_t>(_meshletDrawMetas.size(), 1)
          : 1;
  if (requiredDrawArgumentCount > _drawArgumentCapacity) {
    vkDeviceWaitIdle(_vkContext.device());
    destroyBuffer(_opaqueBucket.drawArgumentBuffer);
    _drawArgumentCapacity = requiredDrawArgumentCount;
    createBuffer(sizeof(DrawIndirectCommand) * _drawArgumentCapacity,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 _opaqueBucket.drawArgumentBuffer);
    recreated = true;
  }

  if (_opaqueBucket.counterBuffer.buffer == VK_NULL_HANDLE) {
    createBuffer(
        sizeof(GpuCounters),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _opaqueBucket.counterBuffer);
    recreated = true;
  }

  if (recreated) {
    updateRenderBucketDescriptorSet();
  }
}

void Renderer::ensureCounterReadbackBuffer(size_t frameIndex) {
  if (frameIndex >= _opaqueBucket.counterReadbackBuffers.size()) {
    _opaqueBucket.counterReadbackBuffers.resize(frameIndex + 1);
    _opaqueBucket.counterReadbackReady.resize(frameIndex + 1, false);
  }

  Buffer &buffer = _opaqueBucket.counterReadbackBuffers[frameIndex];
  if (buffer.buffer != VK_NULL_HANDLE) {
    return;
  }

  createBuffer(sizeof(GpuCounters), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               buffer);
}

void Renderer::readCompletedCounters(size_t frameIndex) {
  if (frameIndex >= _opaqueBucket.counterReadbackBuffers.size() ||
      frameIndex >= _opaqueBucket.counterReadbackReady.size() ||
      !_opaqueBucket.counterReadbackReady[frameIndex]) {
    return;
  }

  const Buffer &buffer = _opaqueBucket.counterReadbackBuffers[frameIndex];
  if (buffer.memory == VK_NULL_HANDLE) {
    return;
  }

  GpuCounters counters{};
  void *data = nullptr;
  vkMapMemory(_vkContext.device(), buffer.memory, 0, sizeof(counters), 0,
              &data);
  std::memcpy(&counters, data, sizeof(counters));
  vkUnmapMemory(_vkContext.device(), buffer.memory);

  _stats.visibleMeshlets = counters.visibleMeshletCount;
  _stats.visibleTriangles = counters.visibleTriangleCount;
}

template <typename T>
void Renderer::uploadDeviceLocalBuffer(const std::vector<T> &source,
                                       VkBufferUsageFlags usage,
                                       Buffer &target) {
  destroyBuffer(target);
  if (source.empty()) {
    return;
  }

  const VkDeviceSize bufferSize = sizeof(source[0]) * source.size();
  Buffer staging;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               staging);

  void *data = nullptr;
  vkMapMemory(_vkContext.device(), staging.memory, 0, bufferSize, 0, &data);
  std::memcpy(data, source.data(), static_cast<size_t>(bufferSize));
  vkUnmapMemory(_vkContext.device(), staging.memory);

  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, target);
  _vkContext.copyBuffer(staging.buffer, target.buffer, bufferSize);
  destroyBuffer(staging);
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties, Buffer &buffer) {
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  const VkResult createBufferResult =
      vkCreateBuffer(_vkContext.device(), &bufferInfo, nullptr, &buffer.buffer);
  assert(createBufferResult == VK_SUCCESS && "failed to create buffer");

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(_vkContext.device(), buffer.buffer,
                                &memRequirements);
  VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      _vkContext.findMemoryType(memRequirements.memoryTypeBits, properties);

  const VkResult allocateBufferMemoryResult = vkAllocateMemory(
      _vkContext.device(), &allocInfo, nullptr, &buffer.memory);
  assert(allocateBufferMemoryResult == VK_SUCCESS &&
         "failed to allocate buffer memory");
  vkBindBufferMemory(_vkContext.device(), buffer.buffer, buffer.memory, 0);
}

void Renderer::destroyBuffer(Buffer &buffer) {
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(_vkContext.device(), buffer.buffer, nullptr);
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(_vkContext.device(), buffer.memory, nullptr);
  }
  buffer = {};
}

void Renderer::cleanupImgui() {
  if (_imguiDescriptorPool == VK_NULL_HANDLE) {
    return;
  }

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  vkDestroyDescriptorPool(_vkContext.device(), _imguiDescriptorPool, nullptr);
  _imguiDescriptorPool = VK_NULL_HANDLE;
}

void Renderer::recordComputeCull(VkCommandBuffer commandBuffer,
                                 VkBuffer counterReadbackBuffer,
                                 const CameraCullData &cullData) {
  if (_candidateMeshlets.empty()) {
    return;
  }

  vkCmdFillBuffer(commandBuffer, _opaqueBucket.counterBuffer.buffer, 0,
                  sizeof(GpuCounters), 0);
  vkCmdFillBuffer(commandBuffer, _opaqueBucket.drawArgumentBuffer.buffer, 0,
                  sizeof(DrawIndirectCommand) * _drawArgumentCapacity, 0);

  std::array<VkBufferMemoryBarrier, 2> transferBarriers{};
  transferBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  transferBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  transferBarriers[0].dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  transferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  transferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  transferBarriers[0].buffer = _opaqueBucket.counterBuffer.buffer;
  transferBarriers[0].size = VK_WHOLE_SIZE;
  transferBarriers[1] = transferBarriers[0];
  transferBarriers[1].buffer = _opaqueBucket.drawArgumentBuffer.buffer;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                       static_cast<uint32_t>(transferBarriers.size()),
                       transferBarriers.data(), 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _computePipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);

  ComputePushConstants pushConstants{};
  pushConstants.view = cullData.view;
  pushConstants.frustum = cullData.frustum;
  pushConstants.zNearFar = cullData.zNearFar;
  pushConstants.candidateCount =
      static_cast<uint32_t>(_candidateMeshlets.size());
  pushConstants.mode = _indirectMode == IndirectMode::MultiDraw ? 1u : 0u;
  pushConstants.meshletDrawCount =
      static_cast<uint32_t>(_meshletDrawMetas.size());
  vkCmdPushConstants(commandBuffer, _pipelineLayout, kPushConstantStages, 0,
                     sizeof(ComputePushConstants), &pushConstants);

  const uint32_t workItemCount =
      maxValue(pushConstants.candidateCount, pushConstants.meshletDrawCount);
  const uint32_t groupCount = (workItemCount + 63u) / 64u;
  vkCmdDispatch(commandBuffer, groupCount, 1, 1);

  std::array<VkBufferMemoryBarrier, 3> computeBarriers{};
  for (VkBufferMemoryBarrier &barrier : computeBarriers) {
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.size = VK_WHOLE_SIZE;
  }
  computeBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  computeBarriers[0].buffer = _opaqueBucket.visibleMeshletBuffer.buffer;
  computeBarriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  computeBarriers[1].buffer = _opaqueBucket.drawArgumentBuffer.buffer;
  computeBarriers[2].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  computeBarriers[2].buffer = _opaqueBucket.counterBuffer.buffer;

  vkCmdPipelineBarrier(
      commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
          VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, nullptr, static_cast<uint32_t>(computeBarriers.size()),
      computeBarriers.data(), 0, nullptr);

  VkBufferCopy counterCopy{};
  counterCopy.size = sizeof(GpuCounters);
  vkCmdCopyBuffer(commandBuffer, _opaqueBucket.counterBuffer.buffer,
                  counterReadbackBuffer, 1, &counterCopy);

  VkBufferMemoryBarrier hostReadBarrier{
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  hostReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  hostReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  hostReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  hostReadBarrier.buffer = counterReadbackBuffer;
  hostReadBarrier.size = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                       &hostReadBarrier, 0, nullptr);
}

void Renderer::renderImgui(VkCommandBuffer commandBuffer, float dt) {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav;
  ImGui::SetNextWindowPos(ImVec2{12.0f, 12.0f}, ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.55f);
  ImGui::Begin("stats", nullptr, flags);
  const float fps = dt > 0.0f ? 1000.0f / dt : 0.0f;
  ImGui::Text("FPS: %.2f", fps);
  ImGui::Text("dt: %.3f ms", dt);
  ImGui::Text("rendered meshlets: %u / %u", _stats.visibleMeshlets,
              _stats.totalMeshlets);
  ImGui::Text("rendered triangles: %u / %llu", _stats.visibleTriangles,
              static_cast<unsigned long long>(_stats.totalTriangles));

  const bool multiDrawSupported = _vkContext.supportsMultiDrawIndirect();
  const char *modeName =
      _indirectMode == IndirectMode::MultiDraw ? "MultiDraw" : "SingleDraw";
  ImGui::Text("indirect mode: %s", modeName);
  if (!multiDrawSupported) {
    ImGui::TextDisabled("multiDrawIndirect unsupported");
  }

  const char *buttonLabel = _indirectMode == IndirectMode::MultiDraw
                                ? "Switch to SingleDraw"
                                : "Switch to MultiDraw";
  const bool disableButton =
      !multiDrawSupported && _indirectMode == IndirectMode::SingleDraw;
  if (disableButton) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button(buttonLabel)) {
    _switchIndirectModeRequested = true;
  }
  if (disableButton) {
    ImGui::EndDisabled();
  }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void Renderer::recordCommandBuffer(const FrameContext &frame,
                                   const glm::mat4 &viewProjection,
                                   const CameraCullData &cullData, float dt) {
  VkCommandBuffer commandBuffer = frame.commandBuffer;
  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  const VkResult beginCommandBufferResult =
      vkBeginCommandBuffer(commandBuffer, &beginInfo);
  assert(beginCommandBufferResult == VK_SUCCESS &&
         "failed to begin recording command buffer");

  recordComputeCull(
      commandBuffer,
      _opaqueBucket.counterReadbackBuffers[frame.frameIndex].buffer, cullData);

  std::array<VkImageMemoryBarrier, 2> beginRenderingBarriers{};
  beginRenderingBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  beginRenderingBarriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  beginRenderingBarriers[0].newLayout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  beginRenderingBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  beginRenderingBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  beginRenderingBarriers[0].image = frame.colorImage;
  beginRenderingBarriers[0].subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_COLOR_BIT;
  beginRenderingBarriers[0].subresourceRange.levelCount = 1;
  beginRenderingBarriers[0].subresourceRange.layerCount = 1;
  beginRenderingBarriers[0].dstAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  beginRenderingBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  beginRenderingBarriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  beginRenderingBarriers[1].newLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  beginRenderingBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  beginRenderingBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  beginRenderingBarriers[1].image = frame.depthImage;
  beginRenderingBarriers[1].subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_DEPTH_BIT;
  beginRenderingBarriers[1].subresourceRange.levelCount = 1;
  beginRenderingBarriers[1].subresourceRange.layerCount = 1;
  beginRenderingBarriers[1].dstAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                       0, 0, nullptr, 0, nullptr,
                       static_cast<uint32_t>(beginRenderingBarriers.size()),
                       beginRenderingBarriers.data());

  VkRenderingAttachmentInfo colorAttachment{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  colorAttachment.imageView = frame.colorImageView;
  colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.clearValue.color = {{0.02f, 0.03f, 0.04f, 1.0f}};

  VkRenderingAttachmentInfo depthAttachment{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAttachment.imageView = frame.depthImageView;
  depthAttachment.imageLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.clearValue.depthStencil = {1.0f, 0};

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
  renderingInfo.renderArea = {{0, 0}, renderTarget.extent};
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments = &colorAttachment;
  renderingInfo.pDepthAttachment = &depthAttachment;

  vkCmdBeginRendering(commandBuffer, &renderingInfo);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    _graphicsPipeline);

  if (!_frameInstances.empty() && _indirectDrawCount > 0) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);
    PushConstants pushConstants{};
    pushConstants.viewProjection = viewProjection;
    vkCmdPushConstants(commandBuffer, _pipelineLayout, kPushConstantStages, 0,
                       sizeof(PushConstants), &pushConstants);
    vkCmdDrawIndirect(commandBuffer, _opaqueBucket.drawArgumentBuffer.buffer, 0,
                      _indirectDrawCount, sizeof(DrawIndirectCommand));
  }

  renderImgui(commandBuffer, dt);
  vkCmdEndRendering(commandBuffer);

  VkImageMemoryBarrier presentBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  presentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  presentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  presentBarrier.image = frame.colorImage;
  presentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  presentBarrier.subresourceRange.levelCount = 1;
  presentBarrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(commandBuffer,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &presentBarrier);

  const VkResult endCommandBufferResult = vkEndCommandBuffer(commandBuffer);
  assert(endCommandBufferResult == VK_SUCCESS &&
         "failed to record command buffer");
}

void Renderer::render(const glm::mat4 &viewProjection,
                      const CameraCullData &cullData, float dt) {
  if (_switchIndirectModeRequested) {
    _switchIndirectModeRequested = false;
    const IndirectMode nextMode = _indirectMode == IndirectMode::MultiDraw
                                      ? IndirectMode::SingleDraw
                                      : IndirectMode::MultiDraw;
    setIndirectMode(nextMode);
  }

  const FrameContext frame = _vkContext.beginFrame();
  ensureCounterReadbackBuffer(frame.frameIndex);
  readCompletedCounters(frame.frameIndex);
  recordCommandBuffer(frame, viewProjection, cullData, dt);
  _opaqueBucket.counterReadbackReady[frame.frameIndex] = true;
  _vkContext.endFrame(frame);
}
