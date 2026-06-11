#define NOMINMAX
#include "renderer.h"

#include "math_helper.h"

#include <array>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <string>

namespace {
constexpr uint32_t kMaxDepthPyramidMips = 16;

struct PushConstants {
  glm::mat4 viewProjection{1.0f};
};

struct ComputePushConstants {
  glm::mat4 view{1.0f};
  glm::vec4 frustum{1.0f};
  glm::vec2 zNearFar{0.01f, 1000.0f};
  glm::vec2 projectionScale{1.0f};
  glm::vec2 hzbSize{1.0f};
  uint32_t candidateCount = 0;
  uint32_t mode = 0;
  uint32_t meshletDrawCount = 0;
  uint32_t hzbMipCount = 0;
  uint32_t hzbEnabled = 0;
  uint32_t padding0 = 0;
};

struct DepthPyramidPushConstants {
  glm::vec2 zNearFar{0.01f, 1000.0f};
  glm::uvec2 srcSize{1, 1};
  glm::uvec2 dstSize{1, 1};
  uint32_t srcMip = 0;
  uint32_t dstMip = 0;
  uint32_t inputIsDepth = 0;
  uint32_t padding0 = 0;
};

static_assert(sizeof(RenderCounters) == 8);

uint32_t packColor(const glm::vec4 &color) {
  const auto packChannel = [](float value) {
    return static_cast<uint32_t>(clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
  };
  return packChannel(color.r) | (packChannel(color.g) << 8) |
         (packChannel(color.b) << 16) | (packChannel(color.a) << 24);
}

std::string formatNumber(uint64_t value) {
  std::string text = std::to_string(value);
  for (int insertPosition = static_cast<int>(text.size()) - 3;
       insertPosition > 0; insertPosition -= 3) {
    text.insert(static_cast<size_t>(insertPosition), ",");
  }
  return text;
}

InstanceData makeInstanceData(const Object3D &object) {
  return {
      .translationScaleX = {object.position.x, object.position.y,
                            object.position.z, object.scale.x},
      .rotation = {object.rotation.x, object.rotation.y, object.rotation.z,
                   object.rotation.w},
      .scaleYZ = {object.scale.y, object.scale.z},
      .packedColor = packColor(object.color),
  };
}

} // namespace

Renderer::Renderer(GLFWwindow *window, std::filesystem::path appDir)
    : _vkContext(window), _resources(std::move(appDir)) {}

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
  createDepthPyramidPipeline();
  createDescriptorPool();
  createDescriptorSet();
  createDepthPyramidResources();
  updateDepthPyramidDescriptorSet();
  for (Buffer<RenderCounters> &buffer : _renderBucket.counterReadbackBuffers) {
    _vkContext.createBuffer(sizeof(RenderCounters),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            buffer);
  }
  createImgui();
}

void Renderer::waitIdle() { _vkContext.waitIdle(); }

void Renderer::cleanup() {
  if (_vkContext.device() != VK_NULL_HANDLE) {
    cleanupImgui();
    for (Buffer<RenderCounters> &buffer :
         _renderBucket.counterReadbackBuffers) {
      _vkContext.destroyBuffer(buffer);
    }
    _vkContext.destroyBuffer(_renderBucket.meshletDrawMetaBuffer);
    _vkContext.destroyBuffer(_renderBucket.drawArgumentBuffer);
    _vkContext.destroyBuffer(_renderBucket.counterBuffer);
    _vkContext.destroyBuffer(_renderBucket.visibleMeshletBuffer);
    _vkContext.destroyBuffer(_renderBucket.candidateMeshletBuffer);
    _vkContext.destroyBuffer(_instanceBuffer);
    _vkContext.destroyBuffer(_clusterTriangleBuffer);
    _vkContext.destroyBuffer(_meshletVertexRefBuffer);
    _vkContext.destroyBuffer(_clusterVertexBuffer);
    _vkContext.destroyBuffer(_meshletBuffer);
    _vkContext.destroyTexture(_depthPyramid);
    vkDestroyPipeline(_vkContext.device(), _depthPyramidPipeline, nullptr);
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
  std::array<VkDescriptorSetLayoutBinding, 13> bindings{};
  for (uint32_t binding = 0; binding < 10; ++binding) {
    bindings[binding] = {
        .binding = binding,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
    };
  }
  bindings[10] = {
      .binding = 10,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  };
  bindings[11] = {
      .binding = 11,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  };
  bindings[12] = {
      .binding = 12,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = kMaxDepthPyramidMips,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  };

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

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
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertShaderModule,
      .pName = "main",
  };

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragShaderModule,
      .pName = "main",
  };
  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkViewport viewport{
      .width = static_cast<float>(renderTarget.extent.width),
      .height = static_cast<float>(renderTarget.extent.height),
      .maxDepth = 1.0f,
  };

  VkRect2D scissor{
      .offset = {0, 0},
      .extent = renderTarget.extent,
  };
  VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
  };

  VkPipelineRasterizationStateCreateInfo rasterizer{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .lineWidth = 1.0f,
  };

  VkPipelineMultisampleStateCreateInfo multisampling{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };

  VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
  };

  VkPipelineColorBlendAttachmentState colorBlendAttachment{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  VkPipelineColorBlendStateCreateInfo colorBlending{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
  };

  VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = static_cast<uint32_t>(maxValue(
          maxValue(sizeof(PushConstants), sizeof(ComputePushConstants)),
          sizeof(DepthPyramidPushConstants))),
  };

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,
  };
  const VkResult createPipelineLayoutResult = vkCreatePipelineLayout(
      _vkContext.device(), &pipelineLayoutInfo, nullptr, &_pipelineLayout);
  assert(createPipelineLayoutResult == VK_SUCCESS &&
         "failed to create pipeline layout");

  VkPipelineRenderingCreateInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &renderTarget.colorFormat,
      .depthAttachmentFormat = renderTarget.depthFormat,
  };

  VkGraphicsPipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &renderingInfo,
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .layout = _pipelineLayout,
      .renderPass = VK_NULL_HANDLE,
      .subpass = 0,
  };

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
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shaderModule,
      .pName = "main",
  };

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = shaderStageInfo,
      .layout = _pipelineLayout,
  };

  const VkResult createComputePipelineResult =
      vkCreateComputePipelines(_vkContext.device(), VK_NULL_HANDLE, 1,
                               &pipelineInfo, nullptr, &_computePipeline);
  assert(createComputePipelineResult == VK_SUCCESS &&
         "failed to create compute pipeline");

  vkDestroyShaderModule(_vkContext.device(), shaderModule, nullptr);
}

void Renderer::createDepthPyramidPipeline() {
  const auto shaderCode = _resources.readBinary("shaders/hzb_reduce.comp.spv");
  const VkShaderModule shaderModule = _vkContext.createShaderModule(shaderCode);

  VkPipelineShaderStageCreateInfo shaderStageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shaderModule,
      .pName = "main",
  };

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = shaderStageInfo,
      .layout = _pipelineLayout,
  };

  const VkResult createComputePipelineResult =
      vkCreateComputePipelines(_vkContext.device(), VK_NULL_HANDLE, 1,
                               &pipelineInfo, nullptr, &_depthPyramidPipeline);
  assert(createComputePipelineResult == VK_SUCCESS &&
         "failed to create depth pyramid pipeline");

  vkDestroyShaderModule(_vkContext.device(), shaderModule, nullptr);
}

void Renderer::createImgui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();

  std::array<VkDescriptorPoolSize, 1> poolSizes{{
      {
          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 16,
      },
  }};

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 16,
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
  };
  const VkResult createPoolResult = vkCreateDescriptorPool(
      _vkContext.device(), &poolInfo, nullptr, &_imguiDescriptorPool);
  assert(createPoolResult == VK_SUCCESS &&
         "failed to create ImGui descriptor pool");

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkPipelineRenderingCreateInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &renderTarget.colorFormat,
      .depthAttachmentFormat = renderTarget.depthFormat,
  };

  ImGui_ImplGlfw_InitForVulkan(_vkContext._window, true);
  ImGui_ImplVulkan_InitInfo initInfo{
      .ApiVersion = VK_API_VERSION_1_3,
      .Instance = _vkContext.instance(),
      .PhysicalDevice = _vkContext.physicalDevice(),
      .Device = _vkContext.device(),
      .QueueFamily = _vkContext.graphicsQueueFamily(),
      .Queue = _vkContext.graphicsQueue(),
      .DescriptorPool = _imguiDescriptorPool,
      .MinImageCount = _vkContext.swapChainImageCount(),
      .ImageCount = _vkContext.swapChainImageCount(),
      .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
      .UseDynamicRendering = true,
      .PipelineRenderingCreateInfo = renderingInfo,
  };
  ImGui_ImplVulkan_Init(&initInfo);
}

void Renderer::createDescriptorPool() {
  std::array<VkDescriptorPoolSize, 3> poolSizes{{
      {
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 10,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 2,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = kMaxDepthPyramidMips,
      },
  }};

  VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
  };

  const VkResult createDescriptorPoolResult = vkCreateDescriptorPool(
      _vkContext.device(), &poolInfo, nullptr, &_descriptorPool);
  assert(createDescriptorPoolResult == VK_SUCCESS &&
         "failed to create descriptor pool");
}

void Renderer::createDescriptorSet() {
  VkDescriptorSetLayout layout = _descriptorSetLayout;
  VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
  };

  const VkResult allocateDescriptorSetResult = vkAllocateDescriptorSets(
      _vkContext.device(), &allocInfo, &_descriptorSet);
  assert(allocateDescriptorSetResult == VK_SUCCESS &&
         "failed to allocate descriptor set");
}

void Renderer::updateDepthPyramidDescriptorSet() {
  VkDescriptorImageInfo hzbTextureInfo{
      .sampler = _depthPyramid.sampler,
      .imageView = _depthPyramid.view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };
  VkDescriptorImageInfo sceneDepthInfo{
      .sampler = _depthPyramid.sampler,
      .imageView = _vkContext.depthImageView(),
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };

  std::array<VkDescriptorImageInfo, kMaxDepthPyramidMips> hzbMipInfos{};
  for (uint32_t mip = 0; mip < hzbMipInfos.size(); ++mip) {
    const uint32_t imageMip =
        mip < _depthPyramid.mipCount ? mip : _depthPyramid.mipCount - 1u;
    hzbMipInfos[mip] = {
        .imageView = _depthPyramid.mipViews[imageMip],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
  }

  std::array<VkWriteDescriptorSet, 3> descriptorWrites{{
      {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _descriptorSet,
          .dstBinding = 10,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &hzbTextureInfo,
      },
      {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _descriptorSet,
          .dstBinding = 11,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &sceneDepthInfo,
      },
      {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _descriptorSet,
          .dstBinding = 12,
          .descriptorCount = kMaxDepthPyramidMips,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = hzbMipInfos.data(),
      },
  }};

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

void Renderer::updateInstanceDescriptorSet() {
  const VkDeviceSize bufferSize =
      sizeof(InstanceData) * _instanceBufferCapacity;

  VkDescriptorBufferInfo bufferInfo{
      .buffer = _instanceBuffer.buffer,
      .offset = 0,
      .range = bufferSize,
  };

  VkWriteDescriptorSet descriptorWrite{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bufferInfo,
  };

  vkUpdateDescriptorSets(_vkContext.device(), 1, &descriptorWrite, 0, nullptr);
}

void Renderer::updateMeshDescriptorSet() {
  assert(!_meshlets.empty());
  assert(!_clusterVertices.empty());
  assert(!_meshletVertexRefs.empty());
  assert(!_packedClusterTriangles.empty());

  std::array<VkDescriptorBufferInfo, 4> bufferInfos{{
      {
          .buffer = _meshletBuffer.buffer,
          .range = sizeof(Meshlet) * _meshlets.size(),
      },
      {
          .buffer = _clusterVertexBuffer.buffer,
          .range = sizeof(MeshletVertex) * _clusterVertices.size(),
      },
      {
          .buffer = _clusterTriangleBuffer.buffer,
          .range = sizeof(uint32_t) * _packedClusterTriangles.size(),
      },
      {
          .buffer = _meshletVertexRefBuffer.buffer,
          .range = sizeof(uint32_t) * _meshletVertexRefs.size(),
      },
  }};

  constexpr std::array<uint32_t, 4> kBindings = {1, 2, 3, 9};
  std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
  for (uint32_t writeIndex = 0; writeIndex < descriptorWrites.size();
       ++writeIndex) {
    descriptorWrites[writeIndex] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _descriptorSet,
        .dstBinding = kBindings[writeIndex],
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos[writeIndex],
    };
  }

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

void Renderer::updateRenderBucketDescriptorSet() {
  assert(_candidateMeshletCapacity > 0);
  assert(_visibleMeshletCapacity > 0);
  assert(_drawArgumentCapacity > 0);
  assert(_meshletDrawMetaCapacity > 0);
  assert(_renderBucket.counterBuffer.buffer != VK_NULL_HANDLE);

  std::array<VkDescriptorBufferInfo, 5> bufferInfos{{
      {
          .buffer = _renderBucket.candidateMeshletBuffer.buffer,
          .range = sizeof(MeshletInstance) * _candidateMeshletCapacity,
      },
      {
          .buffer = _renderBucket.visibleMeshletBuffer.buffer,
          .range = sizeof(MeshletInstance) * _visibleMeshletCapacity,
      },
      {
          .buffer = _renderBucket.drawArgumentBuffer.buffer,
          .range = sizeof(DrawIndirectCommand) * _drawArgumentCapacity,
      },
      {
          .buffer = _renderBucket.counterBuffer.buffer,
          .range = sizeof(RenderCounters),
      },
      {
          .buffer = _renderBucket.meshletDrawMetaBuffer.buffer,
          .range = sizeof(MeshletDrawMeta) * _meshletDrawMetaCapacity,
      },
  }};

  std::array<VkWriteDescriptorSet, 5> descriptorWrites{};
  for (uint32_t writeIndex = 0; writeIndex < descriptorWrites.size();
       ++writeIndex) {
    descriptorWrites[writeIndex] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = _descriptorSet,
        .dstBinding = 4 + writeIndex,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufferInfos[writeIndex],
    };
  }

  vkUpdateDescriptorSets(_vkContext.device(),
                         static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

MeshId Renderer::loadModel(const GltfModel &model) {
  MeshUploadInfo info{
      .firstMeshlet = static_cast<uint32_t>(_meshlets.size()),
      .meshletCount = static_cast<uint32_t>(model.packedMeshlets.size()),
  };
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

void Renderer::setObjects(const std::vector<Object3D> &objects) {
  _frameInstances.clear();
  _candidateMeshlets.clear();
  _meshletDrawMetas.clear();
  uint64_t totalCandidateTriangles = 0;
  _frameInstances.reserve(objects.size());
  std::vector<uint32_t> meshletCandidateCounts(_meshlets.size(), 0);

  for (const Object3D &object : objects) {
    assert(object.meshId < _meshes.size() && "invalid object mesh id");

    const uint32_t instanceId = static_cast<uint32_t>(_frameInstances.size());
    _frameInstances.push_back(makeInstanceData(object));

    const MeshUploadInfo &mesh = _meshes[object.meshId];
    for (uint32_t meshletOffset = 0; meshletOffset < mesh.meshletCount;
         ++meshletOffset) {
      const uint32_t meshletId = mesh.firstMeshlet + meshletOffset;
      const uint32_t triangleCount = meshletTriangleCount(_meshlets[meshletId]);

      _candidateMeshlets.push_back({
          .instanceId = instanceId,
          .meshletId = meshletId,
      });
      ++meshletCandidateCounts[meshletId];
      totalCandidateTriangles += triangleCount;
    }
  }

  _meshletDrawMetas.resize(_meshlets.size());
  uint32_t visibleInstanceOffset = 0;
  for (uint32_t meshletId = 0; meshletId < _meshletDrawMetas.size();
       ++meshletId) {
    _meshletDrawMetas[meshletId] = {
        .visibleInstanceOffset = visibleInstanceOffset,
        .meshletId = meshletId,
    };
    visibleInstanceOffset += meshletCandidateCounts[meshletId];
  }

  _indirectDrawCount = _indirectMode == IndirectMode::MultiDraw
                           ? static_cast<uint32_t>(_meshletDrawMetas.size())
                           : (_candidateMeshlets.empty() ? 0u : 1u);
  _stats.totalMeshlets = static_cast<uint32_t>(_candidateMeshlets.size());
  _stats.totalTriangles = totalCandidateTriangles;

  if (_frameInstances.empty()) {
    return;
  }

  if (_frameInstances.size() > _instanceBufferCapacity) {
    _vkContext.waitIdle();
    _vkContext.destroyBuffer(_instanceBuffer);
    _instanceBufferCapacity = _frameInstances.size();
    const VkDeviceSize capacitySize =
        sizeof(InstanceData) * _instanceBufferCapacity;
    _vkContext.createBuffer(capacitySize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            _instanceBuffer);
    updateInstanceDescriptorSet();
  }

  const VkDeviceSize uploadSize =
      sizeof(_frameInstances[0]) * _frameInstances.size();

  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), _instanceBuffer.allocation, &data);
  std::memcpy(data, _frameInstances.data(), static_cast<size_t>(uploadSize));
  vmaUnmapMemory(_vkContext.allocator(), _instanceBuffer.allocation);

  uploadRenderBucket();
}

template <typename T>
bool Renderer::uploadHostBuffer(const std::vector<T> &source,
                                VkBufferUsageFlags usage, Buffer<T> &target,
                                size_t &capacity) {
  if (source.empty()) {
    return false;
  }

  bool recreated = false;
  const VkDeviceSize bufferSize = sizeof(source[0]) * source.size();
  if (source.size() > capacity) {
    _vkContext.waitIdle();
    _vkContext.destroyBuffer(target);
    capacity = source.size();
    _vkContext.createBuffer(sizeof(source[0]) * capacity, usage,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            target);
    recreated = true;
  }

  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), target.allocation, &data);
  std::memcpy(data, source.data(), static_cast<size_t>(bufferSize));
  vmaUnmapMemory(_vkContext.allocator(), target.allocation);
  return recreated;
}

void Renderer::uploadRenderBucket() {
  bool recreated = uploadHostBuffer(
      _candidateMeshlets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      _renderBucket.candidateMeshletBuffer, _candidateMeshletCapacity);

  recreated |= uploadHostBuffer(
      _meshletDrawMetas, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      _renderBucket.meshletDrawMetaBuffer, _meshletDrawMetaCapacity);

  if (_candidateMeshlets.size() > _visibleMeshletCapacity) {
    _vkContext.waitIdle();
    _vkContext.destroyBuffer(_renderBucket.visibleMeshletBuffer);
    _visibleMeshletCapacity = _candidateMeshlets.size();
    _vkContext.createBuffer(sizeof(MeshletInstance) * _visibleMeshletCapacity,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            _renderBucket.visibleMeshletBuffer);
    recreated = true;
  }

  const size_t requiredDrawArgumentCount =
      _indirectMode == IndirectMode::MultiDraw
          ? maxValue<size_t>(_meshletDrawMetas.size(), 1)
          : 1;
  if (requiredDrawArgumentCount > _drawArgumentCapacity) {
    _vkContext.waitIdle();
    _vkContext.destroyBuffer(_renderBucket.drawArgumentBuffer);
    _drawArgumentCapacity = requiredDrawArgumentCount;
    _vkContext.createBuffer(sizeof(DrawIndirectCommand) * _drawArgumentCapacity,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            _renderBucket.drawArgumentBuffer);
    recreated = true;
  }

  if (_renderBucket.counterBuffer.buffer == VK_NULL_HANDLE) {
    _vkContext.createBuffer(
        sizeof(RenderCounters),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _renderBucket.counterBuffer);
    recreated = true;
  }

  if (recreated) {
    updateRenderBucketDescriptorSet();
  }
}

void Renderer::readCompletedCounters(size_t frameIndex) {
  const Buffer<RenderCounters> &buffer =
      _renderBucket.counterReadbackBuffers[frameIndex];

  RenderCounters counters{};
  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), buffer.allocation, &data);
  std::memcpy(&counters, data, sizeof(counters));
  vmaUnmapMemory(_vkContext.allocator(), buffer.allocation);

  _stats.visibleMeshlets = counters.visibleMeshletCount;
  _stats.visibleTriangles = counters.visibleTriangleCount;
}

template <typename T>
void Renderer::uploadDeviceLocalBuffer(const std::vector<T> &source,
                                       VkBufferUsageFlags usage,
                                       Buffer<T> &target) {
  _vkContext.destroyBuffer(target);
  if (source.empty()) {
    return;
  }

  const VkDeviceSize bufferSize = sizeof(source[0]) * source.size();
  Buffer<T> staging;
  _vkContext.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          staging);

  void *data = nullptr;
  vmaMapMemory(_vkContext.allocator(), staging.allocation, &data);
  std::memcpy(data, source.data(), static_cast<size_t>(bufferSize));
  vmaUnmapMemory(_vkContext.allocator(), staging.allocation);

  _vkContext.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, target);
  _vkContext.copyBuffer(staging.buffer, target.buffer, bufferSize);
  _vkContext.destroyBuffer(staging);
}

void Renderer::createDepthPyramidResources() {
  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  uint32_t mipCount = 1;
  uint32_t mipWidth = renderTarget.extent.width;
  uint32_t mipHeight = renderTarget.extent.height;
  while ((mipWidth > 1 || mipHeight > 1) && mipCount < kMaxDepthPyramidMips) {
    mipWidth = maxValue(mipWidth / 2u, 1u);
    mipHeight = maxValue(mipHeight / 2u, 1u);
    ++mipCount;
  }

  _depthPyramid.width = renderTarget.extent.width;
  _depthPyramid.height = renderTarget.extent.height;
  _depthPyramid.mipCount = mipCount;
  _vkContext.createImage(_depthPyramid.width, _depthPyramid.height,
                         _depthPyramid.mipCount, VK_FORMAT_R32_SFLOAT,
                         VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_STORAGE_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         _depthPyramid.image, _depthPyramid.allocation);

  _depthPyramid.view =
      _vkContext.createImageView(_depthPyramid.image, VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_ASPECT_COLOR_BIT, 0, mipCount);
  _depthPyramid.mipViews.resize(mipCount);
  for (uint32_t mip = 0; mip < mipCount; ++mip) {
    _depthPyramid.mipViews[mip] =
        _vkContext.createImageView(_depthPyramid.image, VK_FORMAT_R32_SFLOAT,
                                   VK_IMAGE_ASPECT_COLOR_BIT, mip, 1);
  }

  VkSamplerCreateInfo samplerInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = static_cast<float>(mipCount - 1),
  };

  const VkResult createSamplerResult = vkCreateSampler(
      _vkContext.device(), &samplerInfo, nullptr, &_depthPyramid.sampler);
  assert(createSamplerResult == VK_SUCCESS && "failed to create sampler");
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
  // reset buffers
  vkCmdFillBuffer(commandBuffer, _renderBucket.counterBuffer.buffer, 0,
                  sizeof(RenderCounters), 0);
  vkCmdFillBuffer(commandBuffer, _renderBucket.drawArgumentBuffer.buffer, 0,
                  sizeof(DrawIndirectCommand) * _drawArgumentCapacity, 0);

  std::array<VkBufferMemoryBarrier, 2> transferBarriers{{
      {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = _renderBucket.counterBuffer.buffer,
          .size = VK_WHOLE_SIZE,
      },
      {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = _renderBucket.drawArgumentBuffer.buffer,
          .size = VK_WHOLE_SIZE,
      },
  }};

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                       static_cast<uint32_t>(transferBarriers.size()),
                       transferBarriers.data(), 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _computePipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);

  ComputePushConstants pushConstants{
      .view = cullData.view,
      .frustum = cullData.frustum,
      .zNearFar = cullData.zNearFar,
      .projectionScale = cullData.projectionScale,
      .hzbSize = {static_cast<float>(_depthPyramid.width),
                  static_cast<float>(_depthPyramid.height)},
      .candidateCount = static_cast<uint32_t>(_candidateMeshlets.size()),
      .mode = _indirectMode == IndirectMode::MultiDraw ? 1u : 0u,
      .meshletDrawCount = static_cast<uint32_t>(_meshletDrawMetas.size()),
      .hzbMipCount = _depthPyramid.mipCount,
      .hzbEnabled = _hzbEnabled && _depthPyramidReady ? 1u : 0u,
  };
  vkCmdPushConstants(commandBuffer, _pipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(ComputePushConstants), &pushConstants);

  const uint32_t workItemCount =
      maxValue(pushConstants.candidateCount, pushConstants.meshletDrawCount);
  const uint32_t groupCount = (workItemCount + 63u) / 64u;
  vkCmdDispatch(commandBuffer, groupCount, 1, 1);

  std::array<VkBufferMemoryBarrier, 3> computeBarriers{{
      {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = _renderBucket.visibleMeshletBuffer.buffer,
          .size = VK_WHOLE_SIZE,
      },
      {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = _renderBucket.drawArgumentBuffer.buffer,
          .size = VK_WHOLE_SIZE,
      },
      {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = _renderBucket.counterBuffer.buffer,
          .size = VK_WHOLE_SIZE,
      },
  }};

  vkCmdPipelineBarrier(
      commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
          VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, nullptr, static_cast<uint32_t>(computeBarriers.size()),
      computeBarriers.data(), 0, nullptr);

  VkBufferCopy counterCopy{
      .size = sizeof(RenderCounters),
  };
  vkCmdCopyBuffer(commandBuffer, _renderBucket.counterBuffer.buffer,
                  counterReadbackBuffer, 1, &counterCopy);

  VkBufferMemoryBarrier hostReadBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = counterReadbackBuffer,
      .size = VK_WHOLE_SIZE,
  };

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
  ImGui::Text("frame time: %.3f ms", dt);
  const std::string visibleMeshlets = formatNumber(_stats.visibleMeshlets);
  const std::string totalMeshlets = formatNumber(_stats.totalMeshlets);
  const std::string visibleTriangles = formatNumber(_stats.visibleTriangles);
  const std::string totalTriangles = formatNumber(_stats.totalTriangles);
  ImGui::Text("rendered meshlets: %s / %s", visibleMeshlets.c_str(),
              totalMeshlets.c_str());
  ImGui::Text("rendered triangles: %s / %s", visibleTriangles.c_str(),
              totalTriangles.c_str());

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
  if (ImGui::Button(buttonLabel)) {
    _switchIndirectModeRequested = true;
  }
  ImGui::Text("HZB occlusion: %s", _hzbEnabled ? "Enabled" : "Disabled");
  const char *hzbButtonLabel = _hzbEnabled ? "Disable HZB" : "Enable HZB";
  if (ImGui::Button(hzbButtonLabel)) {
    _hzbEnabled = !_hzbEnabled;
  }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void Renderer::recordDepthPyramid(VkCommandBuffer commandBuffer,
                                  const FrameContext &frame,
                                  const CameraCullData &cullData) {
  VkImageMemoryBarrier depthReadBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = frame.depthImage,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
              .levelCount = 1,
              .layerCount = 1,
          },
  };

  vkCmdPipelineBarrier(commandBuffer,
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &depthReadBarrier);

  VkImageMemoryBarrier hzbWriteBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = _depthPyramidReady ? VK_ACCESS_SHADER_READ_BIT : 0u,
      .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .oldLayout = _depthPyramidReady ? VK_IMAGE_LAYOUT_GENERAL
                                      : VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = _depthPyramid.image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .levelCount = _depthPyramid.mipCount,
              .layerCount = 1,
          },
  };

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &hzbWriteBarrier);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _depthPyramidPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);

  uint32_t srcWidth = _depthPyramid.width;
  uint32_t srcHeight = _depthPyramid.height;
  for (uint32_t mip = 0; mip < _depthPyramid.mipCount; ++mip) {
    const uint32_t dstWidth = mip == 0 ? srcWidth : maxValue(srcWidth / 2u, 1u);
    const uint32_t dstHeight =
        mip == 0 ? srcHeight : maxValue(srcHeight / 2u, 1u);
    DepthPyramidPushConstants pushConstants{
        .zNearFar = cullData.zNearFar,
        .srcSize = {srcWidth, srcHeight},
        .dstSize = {dstWidth, dstHeight},
        .srcMip = mip == 0 ? 0u : mip - 1u,
        .dstMip = mip,
        .inputIsDepth = mip == 0 ? 1u : 0u,
    };
    vkCmdPushConstants(commandBuffer, _pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(DepthPyramidPushConstants), &pushConstants);

    vkCmdDispatch(commandBuffer, (dstWidth + 7u) / 8u, (dstHeight + 7u) / 8u,
                  1);

    VkImageMemoryBarrier mipBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = _depthPyramid.image,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mip,
                .levelCount = 1,
                .layerCount = 1,
            },
    };

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &mipBarrier);

    srcWidth = dstWidth;
    srcHeight = dstHeight;
  }

  _depthPyramidReady = true;
}

void Renderer::recordRenderingCommands(VkCommandBuffer commandBuffer,
                                       const FrameContext &frame,
                                       const glm::mat4 &viewProjection,
                                       float dt) {
  std::array<VkImageMemoryBarrier, 2> beginRenderingBarriers{{
      {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = frame.colorImage,
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .levelCount = 1,
                  .layerCount = 1,
              },
      },
      {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = frame.depthImage,
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                  .levelCount = 1,
                  .layerCount = 1,
              },
      },
  }};

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                       0, 0, nullptr, 0, nullptr,
                       static_cast<uint32_t>(beginRenderingBarriers.size()),
                       beginRenderingBarriers.data());

  VkRenderingAttachmentInfo colorAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = frame.colorImageView,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.color = {{0.02f, 0.03f, 0.04f, 1.0f}}},
  };

  VkRenderingAttachmentInfo depthAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = frame.depthImageView,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.depthStencil = {.depth = 1.0f, .stencil = 0}},
  };

  const RenderTargetInfo renderTarget = _vkContext.renderTargetInfo();
  VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {.offset = {0, 0}, .extent = renderTarget.extent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachment,
      .pDepthAttachment = &depthAttachment,
  };

  // rendering
  vkCmdBeginRendering(commandBuffer, &renderingInfo);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    _graphicsPipeline);

  if (!_frameInstances.empty() && _indirectDrawCount > 0) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);
    PushConstants pushConstants{
        .viewProjection = viewProjection,
    };
    vkCmdPushConstants(commandBuffer, _pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &pushConstants);
    vkCmdDrawIndirect(commandBuffer, _renderBucket.drawArgumentBuffer.buffer, 0,
                      _indirectDrawCount, sizeof(DrawIndirectCommand));
  }

  renderImgui(commandBuffer, dt);
  vkCmdEndRendering(commandBuffer);

  VkImageMemoryBarrier presentBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = frame.colorImage,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .levelCount = 1,
              .layerCount = 1,
          },
  };

  vkCmdPipelineBarrier(commandBuffer,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &presentBarrier);
}

void Renderer::recordCommandBuffer(const FrameContext &frame,
                                   const glm::mat4 &viewProjection,
                                   const CameraCullData &cullData, float dt) {
  VkCommandBuffer commandBuffer = frame.commandBuffer;
  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  const VkResult beginCommandBufferResult =
      vkBeginCommandBuffer(commandBuffer, &beginInfo);
  assert(beginCommandBufferResult == VK_SUCCESS &&
         "failed to begin recording command buffer");

  recordComputeCull(
      commandBuffer,
      _renderBucket.counterReadbackBuffers[frame.frameIndex].buffer, cullData);
  recordRenderingCommands(commandBuffer, frame, viewProjection, dt);
  recordDepthPyramid(commandBuffer, frame, cullData);

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
  readCompletedCounters(frame.frameIndex);
  recordCommandBuffer(frame, viewProjection, cullData, dt);
  _vkContext.endFrame(frame);
}
