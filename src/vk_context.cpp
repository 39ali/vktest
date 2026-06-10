#define NOMINMAX
#include "vk_context.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <string>

namespace {

const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char *> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

VkResult createDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *createInfo,
    const VkAllocationCallbacks *allocator,
    VkDebugUtilsMessengerEXT *messenger) {
  auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  return fn ? fn(instance, createInfo, allocator, messenger)
            : VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT messenger,
                                   const VkAllocationCallbacks *allocator) {
  auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (fn) {
    fn(instance, messenger, allocator);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *callbackData, void *) {
  std::cerr << "validation: " << callbackData->pMessage << '\n';
  return VK_FALSE;
}

} // namespace

VkContext::VkContext(GLFWwindow *window) : _window(window) {}

VkContext::~VkContext() { cleanup(); }

void VkContext::init() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createCommandPool();
  createCommandBuffers();
  createSwapChain();
  createSyncObjects();
  createDepthResources();
}

void VkContext::cleanup() {
  if (_device != VK_NULL_HANDLE) {
    cleanupSyncObjects();
    cleanupSwapChain();
    if (_commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(_device, _commandPool, nullptr);
      _commandPool = VK_NULL_HANDLE;
    }
    vkDestroyDevice(_device, nullptr);
    _device = VK_NULL_HANDLE;
  }

  if (_instance != VK_NULL_HANDLE) {
    if (kEnableValidationLayers && _debugMessenger != VK_NULL_HANDLE) {
      destroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
      _debugMessenger = VK_NULL_HANDLE;
    }
    if (_surface != VK_NULL_HANDLE) {
      vkDestroySurfaceKHR(_instance, _surface, nullptr);
      _surface = VK_NULL_HANDLE;
    }
    vkDestroyInstance(_instance, nullptr);
    _instance = VK_NULL_HANDLE;
  }
}

void VkContext::cleanupSwapChain() {
  cleanupRenderTargets();

  for (auto imageView : _swapChainImageViews) {
    vkDestroyImageView(_device, imageView, nullptr);
  }
  _swapChainImageViews.clear();
  _swapChainImages.clear();

  if (_swapChain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(_device, _swapChain, nullptr);
    _swapChain = VK_NULL_HANDLE;
  }
}

void VkContext::cleanupRenderTargets() {
  if (_depthImage.view != VK_NULL_HANDLE) {
    vkDestroyImageView(_device, _depthImage.view, nullptr);
  }
  if (_depthImage.image != VK_NULL_HANDLE) {
    vkDestroyImage(_device, _depthImage.image, nullptr);
  }
  if (_depthImage.memory != VK_NULL_HANDLE) {
    vkFreeMemory(_device, _depthImage.memory, nullptr);
  }
  _depthImage = {};
}

void VkContext::cleanupSyncObjects() {
  for (auto semaphore : _renderFinishedSemaphores) {
    vkDestroySemaphore(_device, semaphore, nullptr);
  }
  _renderFinishedSemaphores.clear();

  for (auto semaphore : _imageAvailableSemaphores) {
    vkDestroySemaphore(_device, semaphore, nullptr);
  }
  _imageAvailableSemaphores.clear();

  for (auto fence : _inFlightFences) {
    vkDestroyFence(_device, fence, nullptr);
  }
  _inFlightFences.clear();

  _commandBuffers.clear();
}

void VkContext::waitIdle() const {
  if (_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(_device);
  }
}

void VkContext::createInstance() {
  assert((!kEnableValidationLayers || checkValidationLayerSupport()) &&
         "validation layers requested but not available");

  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "vktest";
  appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  auto extensions = requiredExtensions();
  VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (kEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    createInfo.ppEnabledLayerNames = kValidationLayers.data();
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = &debugCreateInfo;
  }

  const VkResult createInstanceResult =
      vkCreateInstance(&createInfo, nullptr, &_instance);
  assert(createInstanceResult == VK_SUCCESS &&
         "failed to create Vulkan instance");
}

bool VkContext::checkValidationLayerSupport() const {
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  for (const char *layerName : kValidationLayers) {
    const bool found = std::any_of(
        availableLayers.begin(), availableLayers.end(), [&](const auto &layer) {
          return std::strcmp(layerName, layer.layerName) == 0;
        });
    if (!found) {
      return false;
    }
  }
  return true;
}

std::vector<const char *> VkContext::requiredExtensions() const {
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);
  if (kEnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

void VkContext::populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) const {
  createInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
}

void VkContext::setupDebugMessenger() {
  if (!kEnableValidationLayers) {
    return;
  }
  VkDebugUtilsMessengerCreateInfoEXT createInfo{};
  populateDebugMessengerCreateInfo(createInfo);
  const VkResult createDebugMessengerResult = createDebugUtilsMessengerEXT(
      _instance, &createInfo, nullptr, &_debugMessenger);
  assert(createDebugMessengerResult == VK_SUCCESS &&
         "failed to set up debug messenger");
}

void VkContext::createSurface() {
  const VkResult createSurfaceResult =
      glfwCreateWindowSurface(_instance, _window, nullptr, &_surface);
  assert(createSurfaceResult == VK_SUCCESS &&
         "failed to create window surface");
}

void VkContext::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);
  assert(deviceCount > 0 && "failed to find GPUs with Vulkan support");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());
  for (const auto &device : devices) {
    const QueueFamilyIndices indices = findQueueFamilies(device);
    if (isDeviceSuitable(device, indices)) {
      _physicalDevice = device;
      _queueFamilies = indices;
      return;
    }
  }
  assert(false && "failed to find a suitable GPU");
}

bool VkContext::isDeviceSuitable(VkPhysicalDevice device,
                                 const QueueFamilyIndices &indices) const {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);
  if (properties.apiVersion < VK_API_VERSION_1_3) {
    return false;
  }

  const bool extensionsSupported = checkDeviceExtensionSupport(device);
  bool swapChainAdequate = false;
  if (extensionsSupported) {
    const auto swapChainSupport = querySwapChainSupport(device);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }
  return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

VkContext::QueueFamilyIndices
VkContext::findQueueFamilies(VkPhysicalDevice device) const {
  QueueFamilyIndices indices;
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
    }
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &presentSupport);
    if (presentSupport) {
      indices.presentFamily = i;
    }
    if (indices.isComplete()) {
      break;
    }
  }
  return indices;
}

bool VkContext::checkDeviceExtensionSupport(VkPhysicalDevice device) const {
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::set<std::string> requiredExtensions(kDeviceExtensions.begin(),
                                           kDeviceExtensions.end());
  for (const auto &extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }
  return requiredExtensions.empty();
}

VkContext::SwapChainSupportDetails
VkContext::querySwapChainSupport(VkPhysicalDevice device) const {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface,
                                            &details.capabilities);
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &formatCount, nullptr);
  details.formats.resize(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &formatCount,
                                       details.formats.data());
  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &presentModeCount,
                                            nullptr);
  details.presentModes.resize(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &presentModeCount,
                                            details.presentModes.data());
  return details;
}

void VkContext::createLogicalDevice() {
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {
      _queueFamilies.graphicsFamily.value(),
      _queueFamilies.presentFamily.value()};
  const float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceVulkan13Features supportedVulkan13Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceFeatures2 supportedFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  supportedFeatures.pNext = &supportedVulkan13Features;
  vkGetPhysicalDeviceFeatures2(_physicalDevice, &supportedFeatures);
  assert(supportedVulkan13Features.dynamicRendering &&
         "Vulkan 1.3 dynamic rendering is not supported");
  _supportsMultiDrawIndirect =
      supportedFeatures.features.multiDrawIndirect == VK_TRUE;

  VkPhysicalDeviceVulkan13Features vulkan13Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  vulkan13Features.dynamicRendering = VK_TRUE;
  VkPhysicalDeviceFeatures2 deviceFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  deviceFeatures.features.multiDrawIndirect = _supportsMultiDrawIndirect;
  deviceFeatures.pNext = &vulkan13Features;
  VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.pEnabledFeatures = nullptr;
  createInfo.pNext = &deviceFeatures;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(kDeviceExtensions.size());
  createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();
  if (kEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    createInfo.ppEnabledLayerNames = kValidationLayers.data();
  }

  const VkResult createDeviceResult =
      vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device);
  assert(createDeviceResult == VK_SUCCESS && "failed to create logical device");
  vkGetDeviceQueue(_device, _queueFamilies.graphicsFamily.value(), 0,
                   &_graphicsQueue);
  vkGetDeviceQueue(_device, _queueFamilies.presentFamily.value(), 0,
                   &_presentQueue);
}

void VkContext::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = _queueFamilies.graphicsFamily.value();
  const VkResult createCommandPoolResult =
      vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool);
  assert(createCommandPoolResult == VK_SUCCESS &&
         "failed to create command pool");
}

VkSurfaceFormatKHR VkContext::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &formats) const {
  for (const auto &format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return formats[0];
}

VkPresentModeKHR VkContext::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &presentModes) const {
  for (const auto &presentMode : presentModes) {
    if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      return presentMode;
    }
  }

  for (const auto &presentMode : presentModes) {
    if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return presentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VkContext::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR &capabilities) const {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(_window, &width, &height);
  VkExtent2D extent{static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height)};
  extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                            capabilities.maxImageExtent.width);
  extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height);
  return extent;
}

void VkContext::createSwapChain() {
  const auto support = querySwapChainSupport(_physicalDevice);
  const auto surfaceFormat = chooseSwapSurfaceFormat(support.formats);
  const auto presentMode = chooseSwapPresentMode(support.presentModes);
  const auto extent = chooseSwapExtent(support.capabilities);

  uint32_t imageCount = support.capabilities.minImageCount + 1;
  if (support.capabilities.maxImageCount > 0 &&
      imageCount > support.capabilities.maxImageCount) {
    imageCount = support.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  createInfo.surface = _surface;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queueFamilyIndices[] = {_queueFamilies.graphicsFamily.value(),
                                   _queueFamilies.presentFamily.value()};
  if (_queueFamilies.graphicsFamily != _queueFamilies.presentFamily) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = support.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  const VkResult createSwapChainResult =
      vkCreateSwapchainKHR(_device, &createInfo, nullptr, &_swapChain);
  assert(createSwapChainResult == VK_SUCCESS && "failed to create swap chain");

  vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount, nullptr);
  _swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount,
                          _swapChainImages.data());
  _swapChainImageFormat = surfaceFormat.format;
  _swapChainExtent = extent;
  createImageViews();
}

void VkContext::createImageViews() {
  _swapChainImageViews.resize(_swapChainImages.size());
  for (size_t i = 0; i < _swapChainImages.size(); ++i) {
    _swapChainImageViews[i] = createImageView(
        _swapChainImages[i], _swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

VkImageView VkContext::createImageView(VkImage image, VkFormat format,
                                       VkImageAspectFlags aspectFlags) const {
  VkImageViewCreateInfo createInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  createInfo.image = image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format = format;
  createInfo.subresourceRange.aspectMask = aspectFlags;
  createInfo.subresourceRange.baseMipLevel = 0;
  createInfo.subresourceRange.levelCount = 1;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount = 1;

  VkImageView imageView = VK_NULL_HANDLE;
  const VkResult createImageViewResult =
      vkCreateImageView(_device, &createInfo, nullptr, &imageView);
  assert(createImageViewResult == VK_SUCCESS && "failed to create image view");
  return imageView;
}

void VkContext::createDepthResources() {
  createImage(_swapChainExtent.width, _swapChainExtent.height,
              VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _depthImage);
  _depthImage.view = createImageView(_depthImage.image, VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);
}

void VkContext::createCommandBuffers() {
  _commandBuffers.resize(kMaxFramesInFlight);
  VkCommandBufferAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool = _commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(_commandBuffers.size());
  const VkResult allocateCommandBuffersResult =
      vkAllocateCommandBuffers(_device, &allocInfo, _commandBuffers.data());
  assert(allocateCommandBuffersResult == VK_SUCCESS &&
         "failed to allocate command buffers");
}

void VkContext::createSyncObjects() {
  _imageAvailableSemaphores.resize(kMaxFramesInFlight);
  _renderFinishedSemaphores.resize(_swapChainImageViews.size());
  _inFlightFences.resize(kMaxFramesInFlight);

  VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < kMaxFramesInFlight; ++i) {
    const VkResult createImageAvailableSemaphoreResult = vkCreateSemaphore(
        _device, &semaphoreInfo, nullptr, &_imageAvailableSemaphores[i]);
    assert(createImageAvailableSemaphoreResult == VK_SUCCESS &&
           "failed to create image available semaphore");
    const VkResult createFenceResult =
        vkCreateFence(_device, &fenceInfo, nullptr, &_inFlightFences[i]);
    assert(createFenceResult == VK_SUCCESS &&
           "failed to create in-flight fence");
  }

  for (auto &semaphore : _renderFinishedSemaphores) {
    const VkResult createRenderFinishedSemaphoreResult =
        vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &semaphore);
    assert(createRenderFinishedSemaphoreResult == VK_SUCCESS &&
           "failed to create render finished semaphore");
  }
}

FrameContext VkContext::beginFrame() {
  FrameContext frame{};
  frame.frameIndex = _currentFrame;
  frame.commandBuffer = _commandBuffers[_currentFrame];
  frame.depthImage = _depthImage.image;
  frame.depthImageView = _depthImage.view;
  frame.imageAvailableSemaphore = _imageAvailableSemaphores[_currentFrame];
  frame.inFlightFence = _inFlightFences[_currentFrame];

  vkWaitForFences(_device, 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX);

  const VkResult result = vkAcquireNextImageKHR(
      _device, _swapChain, UINT64_MAX, frame.imageAvailableSemaphore,
      VK_NULL_HANDLE, &frame.imageIndex);
  assert(result == VK_SUCCESS && "failed to acquire swap chain image");

  frame.colorImage = _swapChainImages[frame.imageIndex];
  frame.colorImageView = _swapChainImageViews[frame.imageIndex];
  frame.renderFinishedSemaphore = _renderFinishedSemaphores[frame.imageIndex];

  vkResetFences(_device, 1, &frame.inFlightFence);
  vkResetCommandBuffer(frame.commandBuffer, 0);

  return frame;
}

void VkContext::endFrame(const FrameContext &frame) {
  VkSemaphore waitSemaphores[] = {frame.imageAvailableSemaphore};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signalSemaphores[] = {frame.renderFinishedSemaphore};

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  const VkResult queueSubmitResult =
      vkQueueSubmit(_graphicsQueue, 1, &submitInfo, frame.inFlightFence);
  assert(queueSubmitResult == VK_SUCCESS &&
         "failed to submit draw command buffer");

  VkSwapchainKHR swapChains[] = {_swapChain};
  VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &frame.imageIndex;

  const VkResult result = vkQueuePresentKHR(_presentQueue, &presentInfo);
  assert(result == VK_SUCCESS && "failed to present swap chain image");

  _currentFrame = (_currentFrame + 1) % kMaxFramesInFlight;
}

RenderTargetInfo VkContext::renderTargetInfo() const {
  return {_swapChainImageFormat, VK_FORMAT_D32_SFLOAT, _swapChainExtent};
}

void VkContext::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                           VkDeviceSize size) {
  VkCommandBufferAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = _commandPool;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkBufferCopy copyRegion{};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(_graphicsQueue);
  vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
}

VkShaderModule
VkContext::createShaderModule(const std::vector<char> &code) const {
  VkShaderModuleCreateInfo createInfo{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  const VkResult createShaderModuleResult =
      vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule);
  assert(createShaderModuleResult == VK_SUCCESS &&
         "failed to create shader module");
  return shaderModule;
}

void VkContext::createImage(uint32_t width, uint32_t height, VkFormat format,
                            VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties,
                            VkContext::Image &image) const {
  VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  const VkResult createImageResult =
      vkCreateImage(_device, &imageInfo, nullptr, &image.image);
  assert(createImageResult == VK_SUCCESS && "failed to create image");

  VkMemoryRequirements memRequirements{};
  vkGetImageMemoryRequirements(_device, image.image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  const VkResult allocateImageMemoryResult =
      vkAllocateMemory(_device, &allocInfo, nullptr, &image.memory);
  assert(allocateImageMemoryResult == VK_SUCCESS &&
         "failed to allocate image memory");

  vkBindImageMemory(_device, image.image, image.memory, 0);
}

uint32_t VkContext::findMemoryType(uint32_t typeFilter,
                                   VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  assert(false && "failed to find suitable memory type");
  return 0;
}
