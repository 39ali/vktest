#pragma once

#include <optional>
#include <vector>
#include <vulkan/vulkan.h>

struct GLFWwindow;

struct FrameContext {
  size_t frameIndex = 0;
  uint32_t imageIndex = 0;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkImage colorImage = VK_NULL_HANDLE;
  VkImageView colorImageView = VK_NULL_HANDLE;
  VkImage depthImage = VK_NULL_HANDLE;
  VkImageView depthImageView = VK_NULL_HANDLE;
  // graphics queue waits until the swapchain image is available to render into
  VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
  // present queue waits until the graphics queue finished rendering the image
  VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
  // CPU waits until the submitted GPU work for that frame slot is done
  VkFence inFlightFence = VK_NULL_HANDLE;
};

struct RenderTargetInfo {
  VkFormat colorFormat = VK_FORMAT_UNDEFINED;
  VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
  VkExtent2D extent{};
};

class VkContext {
public:
  explicit VkContext(GLFWwindow *window);
  ~VkContext();

  VkContext(const VkContext &) = delete;
  VkContext &operator=(const VkContext &) = delete;

  void init();
  void cleanup();
  FrameContext beginFrame();
  void endFrame(const FrameContext &frame);
  VkShaderModule createShaderModule(const std::vector<char> &code) const;
  RenderTargetInfo renderTargetInfo() const;
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
  void waitIdle() const;

  VkDevice device() const { return _device; }
  VkInstance instance() const { return _instance; }
  VkPhysicalDevice physicalDevice() const { return _physicalDevice; }
  VkQueue graphicsQueue() const { return _graphicsQueue; }
  uint32_t graphicsQueueFamily() const {
    return _queueFamilies.graphicsFamily.value();
  }
  uint32_t swapChainImageCount() const {
    return static_cast<uint32_t>(_swapChainImages.size());
  }
  bool supportsMultiDrawIndirect() const { return _supportsMultiDrawIndirect; }

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) const;

private:
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
  };

  struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
  };

  void createInstance();
  bool checkValidationLayerSupport() const;
  std::vector<const char *> requiredExtensions() const;
  void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo) const;
  void setupDebugMessenger();
  void createSurface();
  void pickPhysicalDevice();
  bool isDeviceSuitable(VkPhysicalDevice device,
                        const QueueFamilyIndices &indices) const;
  bool checkDeviceExtensionSupport(VkPhysicalDevice device) const;
  void createLogicalDevice();
  void createCommandPool();
  void createSwapChain();
  void cleanupSwapChain();
  void createDepthResources();
  void createCommandBuffers();
  void createSyncObjects();
  void cleanupRenderTargets();
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const;
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const;
  VkSurfaceFormatKHR
  chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) const;
  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &presentModes) const;
  VkExtent2D
  chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const;
  void createImageViews();
  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags) const;
  void createImage(uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                   Image &image) const;
  void cleanupSyncObjects();

  static constexpr size_t kMaxFramesInFlight = 2;
  VkInstance _instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;
  VkSurfaceKHR _surface = VK_NULL_HANDLE;
  VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _graphicsQueue = VK_NULL_HANDLE;
  VkQueue _presentQueue = VK_NULL_HANDLE;
  QueueFamilyIndices _queueFamilies;
  VkCommandPool _commandPool = VK_NULL_HANDLE;
  VkSwapchainKHR _swapChain = VK_NULL_HANDLE;
  std::vector<VkImage> _swapChainImages;
  std::vector<VkImageView> _swapChainImageViews;
  VkFormat _swapChainImageFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D _swapChainExtent{};
  Image _depthImage;
  std::vector<VkCommandBuffer> _commandBuffers;
  std::vector<VkSemaphore> _imageAvailableSemaphores;
  std::vector<VkSemaphore> _renderFinishedSemaphores;
  std::vector<VkFence> _inFlightFences;
  size_t _currentFrame = 0;
  bool _supportsMultiDrawIndirect = false;

public:
  GLFWwindow *_window = nullptr;
};
