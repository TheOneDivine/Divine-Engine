#include "App.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "Application/Instance.hpp"
static const std::vector<const char*> deviceExtensions = { "VK_KHR_swapchain" };

/*** Vulkan Setup ***/

void App::initVulkan() {
   m_instance = (VkInstance)GAPIInstance::Get();
   createSurface();//
   pickPhysicalDevice();//
   createLogicalDevice();//
   createSwapChain();//
   createSwapChainImageViews();//
   loadScene();
   createComputeResourceDescriptorSetLayout();
   createComputePipeline();
   createRenderPass();
   createGraphicsResourceDescriptorSetLayout();
   createGraphicsPipeline();//
   createCommandPool();//
   createComputeImage();
   createColorResources();
   createDepthResources();
   createFramebuffers();//
   //createTextureImage();
   createTextureSampler();
   loadQuad();
   createRayTracerSSBOs();
   createRayTracerViewUBO();
   createVertexBuffer();
   createIndexBuffer();
   createGraphicsUBO();
   createPipelineResourceDescriptorSetsPool();//
   allocateComputeResourceDescriptorSets();//
   allocateGraphicsResourceDescriptorSets();//
   createCommandBuffers(m_graphicsCommandBuffers);//
   createCommandBuffers(m_computeCommandBuffers);//
   createSyncObjects();//
} 

/*** GLFW Window ***/

void App::initGLFW() {
   // creates GLFW instance
   glfwInit();
   // enables/disables GLFW features
   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

   // creates window instance and initiates client callbacks
   m_window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, NULL, NULL);
   glfwSetWindowUserPointer(m_window, this);
   glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
   glfwSetKeyCallback(m_window, keyCallback);
   glfwSetCursorPosCallback(m_window, cursorPosCallback);
}
void App::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
   auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));

   if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
      app->m_firstMouse = true;
      return;
   }

   if (app->m_firstMouse) {
      app->m_firstMousePos = { xpos, ypos };
      app->m_firstMouse = false;
      return;
   }

   glm::vec3 direction = glm::normalize(app->m_rayCamera.lookat - app->m_rayCamera.center);
   double yaw = glm::degrees(atan2(direction.z, direction.x));  // Initial yaw (pointing towards negative Z)
   double pitch = glm::degrees(asin(direction.y));  // Initial pitch (level with the horizon)
   const double sensitivity = 20.0f / app->m_swapChainExtent.height;

   glm::vec2 deltaPos = { app->m_firstMousePos.x - xpos, app->m_firstMousePos.y - ypos };

   yaw += deltaPos.x * sensitivity;
   pitch -= deltaPos.y * sensitivity;

   // Constrain pitch to avoid gimbal lock
   if (pitch > 89.0f) pitch = 89.0f;
   if (pitch < -89.0f) pitch = -89.0f;

   glm::vec3 front{};
   front.x = float(cos(glm::radians(yaw)) * cos(glm::radians(pitch)));
   front.y = float(sin(glm::radians(pitch)));
   front.z = float(sin(glm::radians(yaw)) * cos(glm::radians(pitch)));

   app->m_rayCamera.lookat = app->m_rayCamera.center + front;

   app->m_firstMousePos = { xpos, ypos };
}
void App::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
   auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
   glm::vec3 moveDir = { 0.0f, 0.0f, 0.0f };
   // Calculate directions based on the lookAt vector
   glm::vec3 forwardDir = glm::normalize(app->m_rayCamera.lookat - app->m_rayCamera.center); // Forward
   glm::vec3 rightDir = glm::normalize(glm::cross(forwardDir, app->m_rayCamera.vup));        // Right
   glm::vec3 upDir = glm::normalize(app->m_rayCamera.vup);                                   // Up
   const float speed = 0.01f;

   if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { moveDir += forwardDir; }
   if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { moveDir -= forwardDir; }
   if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { moveDir += rightDir; }
   if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { moveDir -= rightDir; }
   if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { moveDir += upDir; }
   if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { moveDir -= upDir; }


   if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_R:
         app->recreateComputePipeline();
         app->m_currentFrame = 0;
         break;
      case GLFW_KEY_SPACE:
         app->m_isRendering = app->m_isRendering ? false : true;
         break;

         // Handle other keys as needed
      default:
         if (_DEBUGGING_ENABLED) std::cout << "Key pressed: " << key << std::endl;
         break;
      }
   }

   /*if (action == GLFW_REPEAT) {
      switch (key) {

            // Handle other keys as needed
         default:
            if (_DEBUGGING_ENABLED) std::cout << "Key held: " << key << std::endl;
            break;
      }
   }

   if (action == GLFW_RELEASE) {
      switch (key) {

         // Handle other keys as needed
      default:
         if (_DEBUGGING_ENABLED) std::cout << "Key released: " << key << std::endl;
         break;
      }
   }*/

   // Normalize and apply the movement
   if (glm::length(moveDir) > std::numeric_limits<float>::epsilon()) {
      app->m_rayCamera.center += glm::normalize(moveDir) * float(speed * app->m_lastFrameTime);
      app->m_rayCamera.lookat += glm::normalize(moveDir) * float(speed * app->m_lastFrameTime);
   }
}
void App::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
   auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
   app->m_framebufferResized = true;
}
void App::createSurface() {
   // useing GLFW creates a Vulkan window surface
   if (glfwCreateWindowSurface(m_instance, m_window, DEFAULT_ALLOCATOR, &m_surface) != VK_SUCCESS)
      throw std::runtime_error("failed to create window surface!");
}
std::vector<const char*> App::getRequiredExtensions() {
   // gets Vulkan extensions for GLFW compatiblity
   uint32_t glfwExtensionCount = 0;
   const char** glfwExtensions;
   glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

   // handle to extensions
   std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

   // enables validation layer extension
   if (_DEBUGGING_ENABLED) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
   return extensions;
}


/*** Physical and Logical Device ***/

void App::pickPhysicalDevice() {
   // get number of available physical devices 
   // if theres no available devices then return 0
   uint32_t deviceCount = 0;
   vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
   if (deviceCount == 0) throw std::runtime_error("failed to find GPUs with Vulkan support!");

   // get physical devices
   // return the best available device
   std::vector<VkPhysicalDevice> devices(deviceCount);
   vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());
   for (const auto& device : devices) {
      if (isDeviceSuitable(device)) {
         m_physicalDevice = device;
         vkGetPhysicalDeviceProperties(m_physicalDevice, &m_physicalDeviceProperties);
         m_msaaSamples = VK_SAMPLE_COUNT_4_BIT;
         //m_msaaSamples = getMaxUsableSampleCount();
         break;
      }
   }

   if (m_physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("failed to find a suitable GPU!");
   if (sizeof(PushConstantData) > m_physicalDeviceProperties.limits.maxPushConstantsSize) throw std::runtime_error("push constant size is greater than max size!");
}
void App::createLogicalDevice() {
   // get queue families
   m_queueFamilyIndices = findQueueFamilies(m_physicalDevice);

   // create queue for each queue family type
   std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
   float queuePriority = 1.0f;
   std::set<uint32_t> uniqueQueueFamilies = {
      m_queueFamilyIndices.graphicsAndComputeFamily.value(),
      m_queueFamilyIndices.presentFamily.value()
   };
   for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueFamilyIndex = queueFamily,
         .queueCount = 1,
         .pQueuePriorities = &queuePriority
      };
      queueCreateInfos.push_back(queueCreateInfo);
   }

   // additional device features
   // for more info visit -> https://docs.vulkan.org/spec/latest/chapters/features.html#features
   VkPhysicalDeviceFeatures deviceFeatures = {
      .robustBufferAccess = VK_FALSE, // ensures buffer accesses are bounds-checked to prevent out-of-bounds errors
      .fullDrawIndexUint32 = VK_FALSE, // allows the use of 32-bit indices in draw calls
      .imageCubeArray = VK_FALSE, // allows the use of 32-bit indices in draw calls
      .independentBlend = VK_FALSE, // allows independent blending operations for each color attachment
      .geometryShader = VK_FALSE, // enables the use of geometry shaders
      .tessellationShader = VK_FALSE, // enables the use of tessellation shaders
      .sampleRateShading = VK_TRUE,  // allows fragment shaders to run at sample rate rather than pixel rate
      .dualSrcBlend = VK_FALSE, // enables dual-source blending
      .logicOp = VK_FALSE, // allows logical operations in the blend stage
      .multiDrawIndirect = VK_FALSE, // enables multiple draw commands to be issued with a single call.
      .drawIndirectFirstInstance = VK_FALSE, // allows the first instance to be specified in indirect draw calls
      .depthClamp = VK_FALSE, // enables depth clamping
      .depthBiasClamp = VK_FALSE, // allows depth bias clamping
      .fillModeNonSolid = VK_FALSE, // enables non-solid fill modes
      .depthBounds = VK_FALSE, // allows depth bounds testing
      .wideLines = VK_FALSE, // enables the use of wide lines
      .largePoints = VK_FALSE, // allows the use of large points
      .alphaToOne = VK_FALSE, // enables alpha-to-one blending
      .multiViewport = VK_FALSE, // allows multiple viewports
      .samplerAnisotropy = VK_TRUE,  // enables anisotropic filtering in samplers
      .textureCompressionETC2 = VK_FALSE, // supports ETC2 texture compression
      .textureCompressionASTC_LDR = VK_FALSE, // supports ASTC LDR texture compression
      .textureCompressionBC = VK_FALSE, // supports BC texture compression
      .occlusionQueryPrecise = VK_FALSE, // enables precise occlusion queries
      .pipelineStatisticsQuery = VK_FALSE, // allows pipeline statistics queries
      .vertexPipelineStoresAndAtomics = VK_FALSE, // enables stores and atomics in the vertex pipeline
      .fragmentStoresAndAtomics = VK_FALSE, // enables stores and atomics in the fragment pipeline
      .shaderTessellationAndGeometryPointSize = VK_FALSE, // allows shaders to specify point sizes in tessellation and geometry stages
      .shaderImageGatherExtended = VK_FALSE, // enables extended image gather operations in shaders
      .shaderStorageImageExtendedFormats = VK_FALSE, // supports extended formats for storage images in shaders
      .shaderStorageImageMultisample = VK_FALSE, // allows multisample storage images in shaders
      .shaderStorageImageReadWithoutFormat = VK_FALSE, // enables reading from storage images without specifying a format
      .shaderStorageImageWriteWithoutFormat = VK_FALSE, // enables writing to storage images without specifying a format
      .shaderUniformBufferArrayDynamicIndexing = VK_FALSE, // allows dynamic indexing of uniform buffer arrays in shaders
      .shaderSampledImageArrayDynamicIndexing = VK_FALSE, // allows dynamic indexing of sampled image arrays in shaders
      .shaderStorageBufferArrayDynamicIndexing = VK_FALSE, // allows dynamic indexing of storage buffer arrays in shaders
      .shaderStorageImageArrayDynamicIndexing = VK_FALSE, // allows dynamic indexing of storage image arrays in shaders
      .shaderClipDistance = VK_FALSE, // enables the use of clip distances in shaders
      .shaderCullDistance = VK_FALSE, // enables the use of cull distances in shaders
      .shaderFloat64 = VK_TRUE,  // supports 64-bit floating-point operations in shaders
      .shaderInt64 = VK_FALSE, // supports 64-bit integer operations in shaders
      .shaderInt16 = VK_FALSE, // supports 16-bit integer operations in shaders
      .shaderResourceResidency = VK_FALSE, // allows shaders to query resource residency
      .shaderResourceMinLod = VK_FALSE, // enables shaders to specify minimum LOD for resources
      .sparseBinding = VK_FALSE, // supports sparse resource binding
      .sparseResidencyBuffer = VK_FALSE, // allows sparse residency for buffers
      .sparseResidencyImage2D = VK_FALSE, // allows sparse residency for 2D images
      .sparseResidencyImage3D = VK_FALSE, // allows sparse residency for 3D images
      .sparseResidency2Samples = VK_FALSE, // supports sparse residency for 2-sample images
      .sparseResidency4Samples = VK_FALSE, // supports sparse residency for 4-sample images
      .sparseResidency8Samples = VK_FALSE, // supports sparse residency for 8-sample images
      .sparseResidency16Samples = VK_FALSE, // supports sparse residency for 16-sample images
      .sparseResidencyAliased = VK_FALSE, // allows aliasing of sparse resources
      .variableMultisampleRate = VK_FALSE, // enables variable multisample rates
      .inheritedQueries = VK_FALSE  // allows queries to be inherited by secondary command buffers
   };

   // create logical device
   VkDeviceCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledLayerCount = 0,
      .ppEnabledLayerNames = nullptr,
      .enabledExtensionCount = (uint32_t)deviceExtensions.size(),
      .ppEnabledExtensionNames = deviceExtensions.data(),
      .pEnabledFeatures = &deviceFeatures,
   };
   if (vkCreateDevice(m_physicalDevice, &createInfo, DEFAULT_ALLOCATOR, &m_logicalDevice) != VK_SUCCESS)
      throw std::runtime_error("failed to create logical device!");

   // get handles for queue types in logical device
   vkGetDeviceQueue(m_logicalDevice, m_queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &m_graphicsQueue);
   vkGetDeviceQueue(m_logicalDevice, m_queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &m_computeQueue);
   vkGetDeviceQueue(m_logicalDevice, m_queueFamilyIndices.presentFamily.value(), 0, &m_presentQueue);
}
bool App::isDeviceSuitable(VkPhysicalDevice device) {
   // get supported queue types and check supported extensions
   QueueFamilyIndices indices = findQueueFamilies(device);
   bool extensionsSupported = checkDeviceExtensionSupport(device);

   // checks if physical device has suitable features
   bool swapChainAdequate = false;
   if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
   }
   VkPhysicalDeviceFeatures supportedFeatures;
   vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

   return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}
VkSampleCountFlagBits App::getMaxUsableSampleCount() {
   VkSampleCountFlags counts = m_physicalDeviceProperties.limits.framebufferColorSampleCounts & m_physicalDeviceProperties.limits.framebufferDepthSampleCounts;

   // returns maximum sample count that the device supports
   // e.g. if device supports 15 samples it cant do 16 but can do the next highest of 8
   if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
   if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
   if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
   if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
   if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
   if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }
   return VK_SAMPLE_COUNT_1_BIT;
}
bool App::checkDeviceExtensionSupport(VkPhysicalDevice device) {
   // get number of supported extensions
   uint32_t extensionCount;
   vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
   // list supported extensions
   std::vector<VkExtensionProperties> availableExtensions(extensionCount);
   vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
   // list required extensions
   std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

   // check for all required extensions
   for (const auto& extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
   }
   // return true if all extensions are supported
   return requiredExtensions.empty();
}
QueueFamilyIndices App::findQueueFamilies(VkPhysicalDevice device) {
   // get the number of queue families
   uint32_t queueFamilyCount = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

   // get each queue families properties
   std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
   vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

   // check each queue family properties for the required queue types and return the proper queue family
   QueueFamilyIndices indices;
   int i = 0;
   for (const auto& queueFamily : queueFamilies) {
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);

      if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && // graphics queue type used in graphics pipeline creation
         (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {  // compute queue type used in general computation
         indices.graphicsAndComputeFamily = i;
      }
      if (presentSupport) indices.presentFamily = i; // presentation queue type used in drawing to the window surface
      if (indices.isComplete()) break;

      i++;
   }
   return indices;
}
void App::createSyncObjects() {
   // makes sure all vectors are equal to the number of frames in flight
   m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
   m_graphicsFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
   m_graphicsInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
   m_computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
   m_computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

   // semaphore creation info
   VkSemaphoreCreateInfo semaphoreInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0
   };
   // fence creation info
   VkFenceCreateInfo fenceInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT
   };

   // creates semaphores and fences for each frame in flight
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // graphics stage
      if (vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, DEFAULT_ALLOCATOR, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
         vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, DEFAULT_ALLOCATOR, &m_graphicsFinishedSemaphores[i]) != VK_SUCCESS ||
         vkCreateFence(m_logicalDevice, &fenceInfo, DEFAULT_ALLOCATOR, &m_graphicsInFlightFences[i]) != VK_SUCCESS)
         throw std::runtime_error("failed to create grpahics synchronization objects for a frame!");
      // compute stage
      if (vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, DEFAULT_ALLOCATOR, &m_computeFinishedSemaphores[i]) != VK_SUCCESS ||
         vkCreateFence(m_logicalDevice, &fenceInfo, DEFAULT_ALLOCATOR, &m_computeInFlightFences[i]) != VK_SUCCESS) {
         throw std::runtime_error("failed to create compute synchronization objects for a frame!");
      }
   }
}


/*** Descriptor Buffers ***/

void App::createDescriptorBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
   // create buffer
   VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr
   };
   if (vkCreateBuffer(m_logicalDevice, &bufferInfo, DEFAULT_ALLOCATOR, &buffer) != VK_SUCCESS)
      throw std::runtime_error("failed to create buffer!");

   // get buffer memory requirements
   VkMemoryRequirements memRequirements;
   vkGetBufferMemoryRequirements(m_logicalDevice, buffer, &memRequirements);

   // allocate memory for buffer
   VkMemoryAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = nullptr,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
   };
   if (vkAllocateMemory(m_logicalDevice, &allocInfo, DEFAULT_ALLOCATOR, &bufferMemory) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate buffer memory!");

   // bind memory to buffer
   vkBindBufferMemory(m_logicalDevice, buffer, bufferMemory, 0);
}
void App::copyDescriptorBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkQueue commandQueue) {
   // command buffer memory allocation info
   VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = m_commandBuffersPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1
   };

   // allocate memory for command buffer
   VkCommandBuffer commandBuffer;
   vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, &commandBuffer);

   // begin recording command to buffer
   VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      .pInheritanceInfo = nullptr
   };
   vkBeginCommandBuffer(commandBuffer, &beginInfo);

   // copy buffer and stop recording to buffer
   VkBufferCopy copyRegion{};
   copyRegion.srcOffset = 0;
   copyRegion.dstOffset = 0;
   copyRegion.size = size;
   vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
   vkEndCommandBuffer(commandBuffer);

   // submit command buffer
   VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr
   };
   vkQueueSubmit(commandQueue, 1, &submitInfo, VK_NULL_HANDLE);
   vkQueueWaitIdle(commandQueue);

   // free command buffer
   vkFreeCommandBuffers(m_logicalDevice, m_commandBuffersPool, 1, &commandBuffer);
}
VkShaderModule App::createShaderModule(const std::vector<uint32_t>& code) {
   // create the shader module
   VkShaderModuleCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .codeSize = sizeof(uint32_t) * code.size(),
      .pCode = (const uint32_t*)code.data()
   };
   VkShaderModule shaderModule;
   if (vkCreateShaderModule(m_logicalDevice, &createInfo, DEFAULT_ALLOCATOR, &shaderModule) != VK_SUCCESS)
      throw std::runtime_error("failed to create shader module!");

   return shaderModule;
}


/*** Descriptor Sets ***/

void App::createPipelineResourceDescriptorSetsPool() {
   // descriptor pool sizes for uniform buffer and texture sampler
   std::array<VkDescriptorPoolSize, 5> poolSizes{};
   // graphic
   poolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         (uint32_t)MAX_FRAMES_IN_FLIGHT };
   poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)MAX_FRAMES_IN_FLIGHT };

   // compute
   poolSizes[2] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         (uint32_t)MAX_FRAMES_IN_FLIGHT };
   poolSizes[3] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          (uint32_t)MAX_FRAMES_IN_FLIGHT };
   poolSizes[4] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         (uint32_t)MAX_FRAMES_IN_FLIGHT };

   // descriptor pool creation info used in creating the descriptor pool object
   VkDescriptorPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = 20,
      .poolSizeCount = (uint32_t)poolSizes.size(),
      .pPoolSizes = poolSizes.data()
   };

   // creates a pool for storage of descriptor sets objects
   if (vkCreateDescriptorPool(m_logicalDevice, &poolInfo, DEFAULT_ALLOCATOR, &m_pipelineDescriptorSetsPool) != VK_SUCCESS)
      throw std::runtime_error("failed to create descriptor pool!");
}

void App::createComputeResourceDescriptorSetLayout() {
   // descriptor set layout bindings for a UBO and two SSBOs
   std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
   // ray tracer viewport UBO
   bindings[0] = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = nullptr
   };
   // ray tracer input/output image
   bindings[1] = {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = nullptr
   };
   // ray spheres SSBO
   bindings[2] = {
      .binding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = nullptr
   };

   // descriptor set layout creation info used in creating the descriptor set layout object
   VkDescriptorSetLayoutCreateInfo layoutInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .bindingCount = (uint32_t)bindings.size(),
      .pBindings = bindings.data()
   };

   // creates a descriptor set layout object
   if (vkCreateDescriptorSetLayout(m_logicalDevice, &layoutInfo, DEFAULT_ALLOCATOR, &m_computeDescriptorSetLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create descriptor set layout!");
}
void App::allocateComputeResourceDescriptorSets() {
   // descriptor set allocation info used in allocating memory in the descriptor pool
   std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_computeDescriptorSetLayout);
   VkDescriptorSetAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = m_pipelineDescriptorSetsPool,
      .descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT,
      .pSetLayouts = layouts.data()
   };

   // allocates memory in descriptor pool for descriptor sets to be stored in
   m_computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
   if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_computeDescriptorSets.data()) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate descriptor sets!");

   // make descriptor writes for each frame in flight
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // descriptor buffer info for uniform buffer and input/output images

      // ray tracer viewport UBO
      VkDescriptorBufferInfo rayViewportInfo = {
         .buffer = m_rayViewPortInfoUBOs[i],
         .offset = 0,
         .range = sizeof(RayViewData)
      };
      // ray tracer input/output image
      VkDescriptorImageInfo rayImageInfo = {
         .sampler = VK_NULL_HANDLE,
         .imageView = m_rayStorageImageView,
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL
      };
      // ray SSBO
      VkDescriptorBufferInfo raySpheres = {
         .buffer = m_raySpheres[i],
         .offset = 0,
         .range = sizeof(m_scene)
      };

      // descriptor writes for uniform buffer to be submited to the descriptor set object
      std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
      // ray tracer viewport UBO
      descriptorWrites[0] = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = nullptr,
         .dstSet = m_computeDescriptorSets[i],
         .dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &rayViewportInfo,
         .pTexelBufferView = nullptr
      };
      // ray tracer input/output image
      descriptorWrites[1] = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = nullptr,
         .dstSet = m_computeDescriptorSets[i],
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &rayImageInfo,
         .pBufferInfo = nullptr,
         .pTexelBufferView = nullptr
      };
      // ray spheres SSBO
      descriptorWrites[2] = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = nullptr,
         .dstSet = m_computeDescriptorSets[i],
         .dstBinding = 2,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &raySpheres,
         .pTexelBufferView = nullptr
      };

      // creates a new descriptor set object
      vkUpdateDescriptorSets(m_logicalDevice, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
   }
}

void App::createGraphicsResourceDescriptorSetLayout() {
   // descriptor set layout bindings for uniform buffer and texture sampler
   std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
   // uniform buffer
   bindings[0] = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .pImmutableSamplers = nullptr
   };
   // combined image sampler
   bindings[1] = {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .pImmutableSamplers = nullptr
   };

   // descriptor set layout creation info used in creating the descriptor set layout object
   VkDescriptorSetLayoutCreateInfo layoutInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .bindingCount = (uint32_t)bindings.size(),
      .pBindings = bindings.data()
   };

   // creates a descriptor set layout object
   if (vkCreateDescriptorSetLayout(m_logicalDevice, &layoutInfo, DEFAULT_ALLOCATOR, &m_graphicsDescriptorSetLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create descriptor set layout!");
}
void App::allocateGraphicsResourceDescriptorSets() {
   // descriptor set allocation info used in allocating memory in the descriptor pool
   std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_graphicsDescriptorSetLayout);
   VkDescriptorSetAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = m_pipelineDescriptorSetsPool,
      .descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT,
      .pSetLayouts = layouts.data()
   };

   // allocates memory in descriptor pool for descriptor sets to be stored in
   m_graphicsDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
   if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_graphicsDescriptorSets.data()) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate descriptor sets!");

   // make descriptor writes for each frame in flight
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // uniform buffer object descriptor
      VkDescriptorBufferInfo bufferInfo = {
         .buffer = m_graphicUBOs[i],
         .offset = 0,
         .range = sizeof(GraphicUBO)
      };

      // image sample descriptor
      VkDescriptorImageInfo imageInfo = {
         .sampler = m_textureSampler,
         .imageView = m_rayStorageImageView,
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
      };

      // descriptor writes for uniform buffer and texture sampler to be submited to the descriptor set object
      std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
      // uniform buffer
      descriptorWrites[0] = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = nullptr,
         .dstSet = m_graphicsDescriptorSets[i],
         .dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &bufferInfo,
         .pTexelBufferView = nullptr
      };
      // image sample
      descriptorWrites[1] = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = nullptr,
         .dstSet = m_graphicsDescriptorSets[i],
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &imageInfo,
         .pBufferInfo = nullptr,
         .pTexelBufferView = nullptr
      };

      // creates a new descriptor set object
      vkUpdateDescriptorSets(m_logicalDevice, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
   }
}


/*** Pipeline Descriptors ***/

void App::updatePushConstants() {
   m_pushConstantData.deltaTime = m_lastFrameTime;
   m_pushConstantData.currentFrame = m_currentFrame;
}

void App::loadScene() {
   switch (0) {
   case 0: // different materials
   {
      m_scene.sphere[0] = {
         .radius = 5.0f,
         .center = { 0.0f, 0.0f, 0.0f },
         .material = {
            .smoothness = 5.0f,
            .refraction = 1.0f,
            .color = { 1.0f, 0.0f, 0.0f },
         },
      };
      m_scene.count = 1;
      /* auto ground = m_scene->createEntity();
       ground.addComponent<SphereComponent>(glm::vec3{ 0.0f, -100.5f, -1.0f }, 100.0f);
       ground.addComponent<MaterialComponent>(glm::vec3{ 0.8f, 0.8f, 0.0f });*/

       /*material ground = {
             .type = LAMBERTIAN,
             .color = { 0.8f, 0.8f, 0.0f }
             };

       m_scene.makeSphere(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f, ground);

       material center = {
       .type = LAMBERTIAN,
       .color = { 0.1f, 0.2f, 0.5f }
       };
       m_scene.makeSphere(glm::vec3(0.0f, 0.0f, -1.2f), 0.5f, center);

       material bubble = {
       .type = DIELECTRIC,
       .refractionIndex = 1.0f / 1.5f
       };
       m_scene.makeSphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.4f, bubble);

       material left = {
       .type = DIELECTRIC,
       .refractionIndex = 1.5f
       };
       m_scene.makeSphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5f, left);

       material right = {
       .type = METAL,
       .color = { 0.8f, 0.6f, 0.2f },
       .smoothness = 1.0f
       };
       m_scene.makeSphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5f, right);*/

      break;
   }
   /*case 1:// field of view
   {
      float R = std::cos(3.14159f / 4.0f);
      material blue = {
         .type = LAMBERTIAN,
         .color = { 0.0f, 0.0f, 1.0f },
      };
      m_scene.makeSphere(glm::vec3(-R, 0.0f, -1.0f), R, blue);

      material red = {
         .type = LAMBERTIAN,
         .color = { 1.0f, 0.0f, 0.0f },
      };
      m_scene.makeSphere(glm::vec3(R, 0.0f, -1.0f), R, red);

      break;
   }
   case 2:// book 1: The Final Render
   {
      // ground
      material ground = {
         .type = LAMBERTIAN,
         .color = { 0.5f, 0.5f, 0.5f },
      };
      m_scene.makeSphere(glm::vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground);

      for (int a = -11; a < 11; a++) {
         for (int b = -11; b < 11; b++) {
            float choose_mat = randomFloat();
            glm::vec3 center(a + 0.9f * randomFloat(), 0.2f, b + 0.9f * randomFloat());

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
               if (choose_mat < 0.8f) {
                  // diffuse
                  glm::vec3 albedo = glm::vec3(randomFloat(), randomFloat(), randomFloat()) * glm::vec3(randomFloat(), randomFloat(), randomFloat());

                  material diffuse = {
                     .type = LAMBERTIAN,
                     .color = albedo
                  };
                  m_scene.makeSphere(center, 0.2f, diffuse);
               }
               else if (choose_mat < 0.95f) {
                  // metal
                  glm::vec3 albedo = glm::vec3(randomFloat(0.5f, 1.0f), randomFloat(0.5f, 1.0f), randomFloat(0.5f, 1.0f));
                  float fuzz = randomFloat(0.0f, 0.5f);

                  material metal = {
                     .type = METAL,
                     .color = albedo,
                     .smoothness = fuzz
                  };
                  m_scene.makeSphere(center, 0.2f, metal);
               }
               else {
                  // glass
                  material glass = {
                     .type = DIELECTRIC,
                     .refractionIndex = 1.5f
                  };
                  m_scene.makeSphere(center, 0.2f, glass);
               }
            }
         }
      }

      // big glass sphere
      material Bglass = {
         .type = DIELECTRIC,
         .refractionIndex = 1.5f
      };
      m_scene.makeSphere(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, Bglass);

      // big brown sphere
      material Bdiffuse = {
         .type = LAMBERTIAN,
         .color = { 0.4f, 0.2f, 0.1f }
      };
      m_scene.makeSphere(glm::vec3(-4.0f, 1.0f, 0.0f), 1.0f, Bdiffuse);

      // big metalic sphere
      material Bmetal = {
         .type = METAL,
         .color = { 0.7f, 0.6f, 0.5f },
         .smoothness = 0.0f
      };
      m_scene.makeSphere(glm::vec3(4.0f, 1.0f, 0.0f), 1.0f, Bmetal);

      break;
   }*/
   }
}
void App::createRayTracerSSBOs() {
   VkDeviceSize bufferSize = sizeof(m_scene);

   // Create a staging buffer used to upload data to the gpu
   VkBuffer stagingBuffer;
   VkDeviceMemory stagingBufferMemory;
   createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

   void* data{};
   vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
   memcpy(data, &m_scene, bufferSize);
   vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

   m_raySpheres.resize(MAX_FRAMES_IN_FLIGHT);
   m_raySpheresMemory.resize(MAX_FRAMES_IN_FLIGHT);

   // Copy initial data to all storage buffers
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // sphere data
      createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_raySpheres[i], m_raySpheresMemory[i]);
      copyDescriptorBuffer(stagingBuffer, m_raySpheres[i], bufferSize, m_computeQueue);
   }

   vkDestroyBuffer(m_logicalDevice, stagingBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, stagingBufferMemory, DEFAULT_ALLOCATOR);
}
void App::createRayTracerViewUBO() {
   // resize the uniform buffer vector to the number of frames in flight
   VkDeviceSize bufferSize = sizeof(RayViewData);
   m_rayViewPortInfoUBOs.resize(MAX_FRAMES_IN_FLIGHT);
   m_rayViewPortInfoUBOMemory.resize(MAX_FRAMES_IN_FLIGHT);
   m_rayViewPortInfoMappedUBOs.resize(MAX_FRAMES_IN_FLIGHT);

   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // create the compute uniform buffer object and map the memory for it
      createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_rayViewPortInfoUBOs[i], m_rayViewPortInfoUBOMemory[i]);
      vkMapMemory(m_logicalDevice, m_rayViewPortInfoUBOMemory[i], 0, bufferSize, 0, &m_rayViewPortInfoMappedUBOs[i]);

      updateRayTracerViewUBO(uint32_t(i));
   }
}
void App::updateRayTracerViewUBO(uint32_t currentImage) {
   // Camera
   glm::vec3 cameraCenter = m_rayCamera.center;
   glm::vec3 lookat = m_rayCamera.lookat;
   glm::vec3 vup = m_rayCamera.vup;
   float FOV = m_rayCamera.fov;
   float defocusAngle = m_rayCamera.defocusAngle;
   float focusDistance = m_rayCamera.defocusDistance;


   float h = std::tan(FOV / 2.0f);
   float viewportHeight = 2.0f * h * focusDistance;
   float viewportWidth = viewportHeight * (float(m_swapChainExtent.width) / m_swapChainExtent.height);

   // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
   glm::vec3 w = cameraCenter - lookat;
   w /= glm::length(w);
   glm::vec3 u = cross(vup, w);
   u /= glm::length(u);
   glm::vec3 v = cross(w, u);

   // Calculate the vectors across the horizontal and down the vertical viewport edges.
   glm::vec3 viewportUSpan = viewportWidth * u;
   glm::vec3 viewportVSpan = viewportHeight * -v;

   // Calculate the horizontal and vertical delta vectors from pixel to pixel.
   glm::vec3 pixelDeltaU = viewportUSpan / float(m_swapChainExtent.width);
   glm::vec3 pixelDeltaV = viewportVSpan / float(m_swapChainExtent.height);

   // Calculate the location of the upper left pixel.
   glm::vec3 upperLeftCorner = cameraCenter - (focusDistance * w) - (viewportUSpan / 2.0f) - (viewportVSpan / 2.0f);
   glm::vec3 firstPixel = upperLeftCorner + 0.5f * (pixelDeltaU + pixelDeltaV);

   float defocusRadius = focusDistance * std::tan(glm::radians(defocusAngle / 2.0f));
   glm::vec3 defocusDiskU = u * defocusRadius;
   glm::vec3 defocusDiskV = v * defocusRadius;

   // update the uniform buffer object members
   RayViewData ubo = {
      .winExtent = { m_swapChainExtent.width, m_swapChainExtent.height },
      .firstPixel = firstPixel,
      .pixelDeltaU = pixelDeltaU,
      .pixelDeltaV = pixelDeltaV,
      .defocusDiskU = defocusDiskU,
      .defocusDiskV = defocusDiskV,
      .defocusAngle = defocusAngle,
      .cameraCenter = cameraCenter,
      .isRendering = m_isRendering,
   };

   // copy the uniform buffer object data to the mapped memory
   memcpy(m_rayViewPortInfoMappedUBOs[currentImage], &ubo, sizeof(ubo));
}

void App::createVertexBuffer() {


   VkDeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();

   // create a staging buffer for the vertex data
   VkBuffer stagingBuffer;
   VkDeviceMemory stagingBufferMemory;
   createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

   // copy the vertex data to the staging buffer
   void* data;
   vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
   memcpy(data, m_vertices.data(), (size_t)bufferSize);
   vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

   // create buffer and map vertices to it
   createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertexBuffer, m_vertexBufferMemory);
   copyDescriptorBuffer(stagingBuffer, m_vertexBuffer, bufferSize, m_graphicsQueue);

   // clean up the staging buffer
   vkDestroyBuffer(m_logicalDevice, stagingBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, stagingBufferMemory, DEFAULT_ALLOCATOR);
}
void App::createIndexBuffer() {
   VkDeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();

   // create a staging buffer for the index data
   VkBuffer stagingBuffer;
   VkDeviceMemory stagingBufferMemory;
   createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

   // copy the index data to the staging buffer
   void* data;
   vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
   memcpy(data, m_indices.data(), (size_t)bufferSize);
   vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

   // create buffer and map indecies to it
   createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_indexBuffer, m_indexBufferMemory);
   copyDescriptorBuffer(stagingBuffer, m_indexBuffer, bufferSize, m_graphicsQueue);

   // clean up the staging buffer
   vkDestroyBuffer(m_logicalDevice, stagingBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, stagingBufferMemory, DEFAULT_ALLOCATOR);
}
void App::createGraphicsUBO() {
   // resize the uniform buffer vector to the number of frames in flight
   VkDeviceSize bufferSize = sizeof(GraphicUBO);
   m_graphicUBOs.resize(MAX_FRAMES_IN_FLIGHT);
   m_graphicUBOMemory.resize(MAX_FRAMES_IN_FLIGHT);
   m_graphicMappedUBOs.resize(MAX_FRAMES_IN_FLIGHT);

   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // create the graphics uniform buffer object and map the memory for it
      createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_graphicUBOs[i], m_graphicUBOMemory[i]);
      vkMapMemory(m_logicalDevice, m_graphicUBOMemory[i], 0, bufferSize, 0, &m_graphicMappedUBOs[i]);
   }
}
void App::updateGraphicUBO(uint32_t currentImage) {
   // update the uniform buffer object members
   GraphicUBO ubo{};

   // copy the uniform buffer object data to the mapped memory
   memcpy(m_graphicMappedUBOs[currentImage], &ubo, sizeof(ubo));
}


void App::loadModel() {
   tinyobj::attrib_t attrib;
   std::vector<tinyobj::shape_t> shapes;
   std::vector<tinyobj::material_t> materials;
   std::string warn, err;

   if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
      throw std::runtime_error(warn + err);

   std::unordered_map<Vertex, uint32_t> uniqueVertices{};

   for (const auto& shape : shapes) {
      for (const auto& index : shape.mesh.indices) {
         Vertex vertex{};
         vertex.pos = {
               attrib.vertices[3 * index.vertex_index + 0],
               attrib.vertices[3 * index.vertex_index + 1],
               attrib.vertices[3 * index.vertex_index + 2]
         };

         vertex.texCoord = {
               attrib.texcoords[2 * index.texcoord_index + 0],
               1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
         };

         if (uniqueVertices.count(vertex) == 0) {
            uniqueVertices[vertex] = static_cast<uint32_t>(m_vertices.size());
            m_vertices.push_back(vertex);
         }

         m_indices.push_back(uniqueVertices[vertex]);
      }
   }
}

void App::loadQuad() {
   // Define the vertices for the quad
   std::vector<Vertex> vertices = {
       {{ -1.0f, -1.0f, 1.0 }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }},
       {{  1.0f, -1.0f, 1.0 }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }},
       {{  1.0f,  1.0f, 1.0 }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }},
       {{ -1.0f,  1.0f, 1.0 }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }}
   };

   // Define the indices for the quad
   std::vector<uint16_t> indices = { 0, 3, 2, 2, 1, 0 };

   // Create an unordered_map to handle unique vertices, though in this case, all vertices are unique
   std::unordered_map<Vertex, uint32_t> uniqueVertices{};

   // Clear the existing vertex and index buffers
   m_vertices.clear();
   m_indices.clear();

   for (const auto& vertex : vertices) {
      if (uniqueVertices.count(vertex) == 0) {
         uniqueVertices[vertex] = (uint32_t)m_vertices.size();
         m_vertices.push_back(vertex);
      }
   }

   for (const auto& index : indices) {
      m_indices.push_back(index);
   }
}

/*** Image Buffers ***/

void App::create2DImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
   // create image information for creating and allocating memory for the image
   VkImageCreateInfo imageInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = { width, height, 1 },
      .mipLevels = mipLevels,
      .arrayLayers = 1,
      .samples = numSamples,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
   };
   if (vkCreateImage(m_logicalDevice, &imageInfo, DEFAULT_ALLOCATOR, &image) != VK_SUCCESS)
      throw std::runtime_error("failed to create image!");

   // get memory requirements for the image
   VkMemoryRequirements memRequirements;
   vkGetImageMemoryRequirements(m_logicalDevice, image, &memRequirements);

   // allocate memory for the image
   VkMemoryAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = nullptr,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
   };
   if (vkAllocateMemory(m_logicalDevice, &allocInfo, DEFAULT_ALLOCATOR, &imageMemory) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate image memory!");

   // bind the image to the image memory
   vkBindImageMemory(m_logicalDevice, image, imageMemory, 0);
}
VkImageView App::create2DImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
   // create an image view format
   VkImageViewCreateInfo viewInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
      .subresourceRange = {
         .aspectMask = aspectFlags,
         .baseMipLevel = 0,
         .levelCount = mipLevels,
         .baseArrayLayer = 0,
         .layerCount = 1
      }
   };
   VkImageView imageView;
   if (vkCreateImageView(m_logicalDevice, &viewInfo, DEFAULT_ALLOCATOR, &imageView) != VK_SUCCESS)
      throw std::runtime_error("failed to create image view!");

   return imageView;
}
uint32_t App::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
   VkPhysicalDeviceMemoryProperties memProperties;
   vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

   for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      // check if the memory type is suitable for the image
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
         return i;
      }
   }

   throw std::runtime_error("failed to find suitable memory type!");
}
VkCommandBuffer App::beginSingleTimeCommands() {
   // buffer allocation info
   VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = m_commandBuffersPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1
   };

   // allocate the command buffer
   VkCommandBuffer commandBuffer;
   vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, &commandBuffer);

   // begin the command buffer
   VkCommandBufferBeginInfo beginInfo{};
   beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   beginInfo.pNext = nullptr;
   beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   beginInfo.pInheritanceInfo = nullptr;
   vkBeginCommandBuffer(commandBuffer, &beginInfo);

   // return the new command buffer for recording to and later submiting from
   return commandBuffer;
}
void App::endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue commandQueue) {
   // end the command buffer
   vkEndCommandBuffer(commandBuffer);

   // submit info for new command submition to buffer
   VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr
   };

   // submit the command buffer
   vkQueueSubmit(commandQueue, 1, &submitInfo, VK_NULL_HANDLE);
   vkQueueWaitIdle(commandQueue);

   // clean up the command
   vkFreeCommandBuffers(m_logicalDevice, m_commandBuffersPool, 1, &commandBuffer);
}
void App::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, VkQueue commandQueue) {
   // begin recording the command buffer
   VkCommandBuffer commandBuffer = beginSingleTimeCommands();

   // image memory barrier used for layout transition management
   VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = 0,
      .dstAccessMask = 0,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = 0,
         .levelCount = mipLevels,
         .baseArrayLayer = 0,
         .layerCount = 1
      }
   };

   VkPipelineStageFlags srcStage;
   VkPipelineStageFlags dstStage;

   // find the oppropriate layout transition for the image

   // used when an image is initially created and is being prepared to receive data
   if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
   }
   else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
   }
   else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
   }
   else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
      barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

      srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
   }
   else { throw std::invalid_argument("unsupported layout transition!"); }

   // sets the image memory barrier
   vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

   // stop recording the command buffer
   endSingleTimeCommands(commandBuffer, commandQueue);
}
void App::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, VkQueue commandQueue) {
   // begin recording the command to buffer
   VkCommandBuffer commandBuffer = beginSingleTimeCommands();

   // specifications for copying buffer layout to image
   VkBufferImageCopy region = {
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .imageOffset = { 0, 0, 0 },
      .imageExtent = { width, height, 1 }
   };

   vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

   // stop recording the command to buffer
   endSingleTimeCommands(commandBuffer, commandQueue);
}

void App::generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, VkQueue commandQueue) {
   // checks format properties for linear blitting
   VkFormatProperties formatProperties;
   vkGetPhysicalDeviceFormatProperties(m_physicalDevice, imageFormat, &formatProperties);
   if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
      throw std::runtime_error("texture image format does not support linear blitting!");

   // begin recording the command buffer
   VkCommandBuffer commandBuffer = beginSingleTimeCommands();

   // image memory barrier used for layout transition management
   VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = 1,
         .levelCount = 1,
         .baseArrayLayer = 0,
         .layerCount = 1
      }
   };

   int32_t mipWidth = texWidth;
   int32_t mipHeight = texHeight;

   // loop over image/mipmaps and create subsequent mipmap image barriers
   for (uint32_t i = 1; i < mipLevels; i++) {

      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.subresourceRange.baseMipLevel = i - 1;

      // create a transition barrier for the previous mip level to the current mip level
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

      // image blitting info, describes how the image is blitted
      VkImageBlit blit = {
         .srcSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = i - 1,
            .baseArrayLayer = 0,
            .layerCount = 1
         },
         .srcOffsets = {
            { 0, 0, 0 },
            { mipWidth, mipHeight, 1 },
         },
         .dstSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = i,
            .baseArrayLayer = 0,
            .layerCount = 1
         },
         .dstOffsets = {
            { 0, 0, 0 },
            { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 }
         }
      };

      // previous mip level/image is copied and shrunken by a scale of 1/2 (4x4 -> 2x2, 4x2 -> 2x1, etc.)

      vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      // create a transition barrier for the current mip level to the previous mip level
      // image should increase in size by 1.5 times (previous image + previous image / 2) in the largest dimension (usually width) while the other dimension (usually height) should remain the same
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

      if (mipWidth > 1) mipWidth /= 2;
      if (mipHeight > 1) mipHeight /= 2;
   }

   barrier.subresourceRange.baseMipLevel = mipLevels - 1;
   barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

   // create a transition barrier for the last mip level to the command buffer
   vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

   // stops recording the command buffer
   endSingleTimeCommands(commandBuffer, commandQueue);
}

VkFormat App::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
   for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);

      // checks if the format supports the specified tiling and features
      if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) return format;
      else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return format;
   }

   throw std::runtime_error("failed to find supported format!");
}


/*** Pipeline Images ***/

void App::createComputeImage() {
   VkDeviceSize imageSize = m_swapChainExtent.width * m_swapChainExtent.height * 4;

   // creates an intermidiate staging buffer
   VkBuffer stagingBuffer;
   VkDeviceMemory stagingBufferMemory;
   createDescriptorBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

   // copies raw pixel data to staging buffer
   void* data;
   vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
   memset(data, 0, (size_t)imageSize);
   vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

   // Input Image //

   std::vector<VkFormat> computeCandidates = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8_UNORM, VK_FORMAT_R16_SFLOAT };
   VkFormat imageFormat = findSupportedFormat(computeCandidates, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

   // creates a Vulkan image object
   create2DImage(m_swapChainExtent.width, m_swapChainExtent.height, 1, VK_SAMPLE_COUNT_1_BIT, imageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_rayStorageImage, m_rayStorageImageMemory);
   // transitions the staging buffer into the proper layout and copies the staging buffer into the Vulkan image object
   transitionImageLayout(m_rayStorageImage, imageFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, m_computeQueue);
   copyBufferToImage(stagingBuffer, m_rayStorageImage, (uint32_t)m_swapChainExtent.width, (uint32_t)m_swapChainExtent.height, m_computeQueue);

   transitionImageLayout(m_rayStorageImage, imageFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, 1, m_computeQueue);

   // initialize image view
   m_rayStorageImageView = create2DImageView(m_rayStorageImage, imageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

   // destroys the staging buffer
   vkDestroyBuffer(m_logicalDevice, stagingBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, stagingBufferMemory, DEFAULT_ALLOCATOR);
}
void App::recreateComputeImage() {
   VkFenceCreateInfo fenceCreateInfo = {};
   fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
   VkFence computeFence, graphicsFence;
   vkCreateFence(m_logicalDevice, &fenceCreateInfo, DEFAULT_ALLOCATOR, &computeFence);
   vkCreateFence(m_logicalDevice, &fenceCreateInfo, DEFAULT_ALLOCATOR, &graphicsFence);

   // Submit command buffer to compute queue and wait with a fence
   vkQueueSubmit(m_computeQueue, 0, nullptr, computeFence);

   // Submit command buffer to graphics queue and wait with a fence
   vkQueueSubmit(m_graphicsQueue, 0, nullptr, graphicsFence);

   // Wait for fences to ensure all operations are complete
   vkWaitForFences(m_logicalDevice, 1, &computeFence, VK_TRUE, UINT64_MAX);
   vkWaitForFences(m_logicalDevice, 1, &graphicsFence, VK_TRUE, UINT64_MAX);

   // Destroy the fences as they are no longer needed
   vkDestroyFence(m_logicalDevice, computeFence, DEFAULT_ALLOCATOR);
   vkDestroyFence(m_logicalDevice, graphicsFence, DEFAULT_ALLOCATOR);

   vkDestroyImageView(m_logicalDevice, m_rayStorageImageView, DEFAULT_ALLOCATOR);
   vkDestroyImage(m_logicalDevice, m_rayStorageImage, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, m_rayStorageImageMemory, DEFAULT_ALLOCATOR);

   vkDestroyDescriptorPool(m_logicalDevice, m_pipelineDescriptorSetsPool, DEFAULT_ALLOCATOR);

   createComputeImage();

   createPipelineResourceDescriptorSetsPool();

   allocateComputeResourceDescriptorSets();
   allocateGraphicsResourceDescriptorSets();
}

void App::createTextureImage() {
   // load the image into a raw RGBA pixel format and check if not success
   // 1 byte per each red, blue, green, and alpha channel in an linear array of pixels
   int texWidth, texHeight, texChannels;
   stbi_uc* pixels;
   if (!(pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha)))
      throw std::runtime_error("failed to load texture image!");
   VkDeviceSize imageSize = texWidth * texHeight * 4;

   // set mipmap level count for current image texture
   //m_mipLevels = (uint32_t)std::floor(std::log2(std::max(texWidth, texHeight))) + 1;
   m_mipLevels = 1;

   // creates an intermidiate staging buffer
   VkBuffer stagingBuffer;
   VkDeviceMemory stagingBufferMemory;
   createDescriptorBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

   // copies raw pixel data to staging buffer
   void* data;
   vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
   memcpy(data, pixels, (size_t)imageSize);
   vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

   // free the raw pixel data
   stbi_image_free(pixels);

   // creates a Vulkan image object
   create2DImage(texWidth, texHeight, m_mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_rayStorageImage, m_rayStorageImageMemory);

   // transitions the staging buffer into the proper layout and copies the staging buffer into the Vulkan image object
   transitionImageLayout(m_rayStorageImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, m_mipLevels, m_graphicsQueue);
   copyBufferToImage(stagingBuffer, m_rayStorageImage, (uint32_t)texWidth, (uint32_t)texHeight, m_graphicsQueue);

   // initialize image view
   m_rayStorageImageView = create2DImageView(m_rayStorageImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, m_mipLevels);

   // destroys the staging buffer
   vkDestroyBuffer(m_logicalDevice, stagingBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, stagingBufferMemory, DEFAULT_ALLOCATOR);

   // generates mipmap images from the Vulkan image object
   generateMipmaps(m_rayStorageImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, m_mipLevels, m_graphicsQueue);
}
void App::createTextureSampler() {
   // create a texture sampler object
   VkSamplerCreateInfo samplerInfo = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_TRUE,
      .maxAnisotropy = m_physicalDeviceProperties.limits.maxSamplerAnisotropy,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .minLod = 0.0f,
      .maxLod = VK_LOD_CLAMP_NONE,
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE
   };
   if (vkCreateSampler(m_logicalDevice, &samplerInfo, DEFAULT_ALLOCATOR, &m_textureSampler) != VK_SUCCESS)
      throw std::runtime_error("failed to create texture sampler!");
}
void App::createColorResources() {
   VkFormat colorFormat = m_swapChainImageFormat;

   // allocate memory for color image and image view buffers
   create2DImage(m_swapChainExtent.width, m_swapChainExtent.height, 1, m_msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_colorImage, m_colorImageMemory);
   m_colorImageView = create2DImageView(m_colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}
void App::createDepthResources() {
   // finds a supported depth and stencil format
   VkFormat depthFormat = findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

   // allocate memory for depth image and image view buffers
   create2DImage(m_swapChainExtent.width, m_swapChainExtent.height, 1, m_msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthImage, m_depthImageMemory);
   m_depthImageView = create2DImageView(m_depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
}


/*** Swap Chain ***/

void App::createSwapChain() {
   // get supported swap chain properties
   SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);

   // set the most optimal swap chain features
   VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
   VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
   VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

   // get max number of frames device can render at once
   uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
   if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) imageCount = swapChainSupport.capabilities.maxImageCount;

   // create swap chain
   uint32_t queueFamilyIndices[] = { m_queueFamilyIndices.graphicsAndComputeFamily.value(), m_queueFamilyIndices.presentFamily.value() };
   VkSwapchainCreateInfoKHR createInfo = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = 0,
      .surface = m_surface,
      .minImageCount = imageCount,
      .imageFormat = surfaceFormat.format,
      .imageColorSpace = surfaceFormat.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = m_queueFamilyIndices.graphicsAndComputeFamily != m_queueFamilyIndices.presentFamily ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 2,
      .pQueueFamilyIndices = queueFamilyIndices,
      .preTransform = swapChainSupport.capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = presentMode,
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE,
   };
   if (vkCreateSwapchainKHR(m_logicalDevice, &createInfo, DEFAULT_ALLOCATOR, &m_swapChain) != VK_SUCCESS)
      throw std::runtime_error("failed to create swap chain!");

   // get number of swap chains images data needed and set data into swap chain image data
   vkGetSwapchainImagesKHR(m_logicalDevice, m_swapChain, &imageCount, nullptr);
   m_swapChainImages.resize(imageCount);
   vkGetSwapchainImagesKHR(m_logicalDevice, m_swapChain, &imageCount, m_swapChainImages.data());
   m_swapChainImageFormat = surfaceFormat.format;
   m_swapChainExtent = extent;
}
void App::recreateSwapChain() {
   // get the new window dimensions
   int width = 0, height = 0;
   while (width == 0 || height == 0) {
      glfwGetFramebufferSize(m_window, &width, &height);
      glfwWaitEvents();
   }
   // stop rendering until window is resized
   vkDeviceWaitIdle(m_logicalDevice);

   // clean up old swap chain resources
   cleanupSwapChain();

   // create new swap chain resources
   createSwapChain();
   createSwapChainImageViews();
   createColorResources();
   createDepthResources();
   createFramebuffers();
}
void App::cleanupSwapChain() {
   // free up depth resources
   vkDestroyImageView(m_logicalDevice, m_depthImageView, DEFAULT_ALLOCATOR);
   vkDestroyImage(m_logicalDevice, m_depthImage, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, m_depthImageMemory, DEFAULT_ALLOCATOR);
   // free up color resources
   vkDestroyImageView(m_logicalDevice, m_colorImageView, DEFAULT_ALLOCATOR);
   vkDestroyImage(m_logicalDevice, m_colorImage, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, m_colorImageMemory, DEFAULT_ALLOCATOR);
   // destroy frame buffers
   for (auto framebuffer : m_swapChainFramebuffers) {
      vkDestroyFramebuffer(m_logicalDevice, framebuffer, DEFAULT_ALLOCATOR);
   }
   // destroy image views
   for (auto imageView : m_swapChainImageViews) {
      vkDestroyImageView(m_logicalDevice, imageView, DEFAULT_ALLOCATOR);
   }
   // destroy swap chain
   vkDestroySwapchainKHR(m_logicalDevice, m_swapChain, DEFAULT_ALLOCATOR);
}

SwapChainSupportDetails App::querySwapChainSupport(VkPhysicalDevice device) {
   // get the surface capability details of the physical device
   SwapChainSupportDetails details;
   vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &details.capabilities);

   // get the number of surface formats supportedby the physical device
   uint32_t formatCount;
   vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);

   // get the number of present modes supported by the physical 
   uint32_t presentModeCount;
   vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);

   // get the surface formats supported by the physical device
   if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, details.formats.data());
   }

   // get the present modes supported by the physical device
   if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, details.presentModes.data());
   }

   // return all the collected swap chain surface details
   return details;
}
VkSurfaceFormatKHR App::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
   // determines which surface format (data orginization) and color space to use, preferably use the VK_FORMAT_R8G8B8A8_UNORM format
   // and VK_COLOR_SPACE_SRGB_NONLINEAR_KHR color space (gamma correction)
   for (const auto& availableFormat : availableFormats) { if (availableFormat.format == VK_FORMAT_R8G8B8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return availableFormat; }


   // returns the first available format if the preferred format is not available
   return availableFormats[0];
}
VkPresentModeKHR App::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
   // determines which present mode (vertical sync method) to use, preferably use the VK_PRESENT_MODE_MAILBOX_KHR mode
   for (const auto& availablePresentMode : availablePresentModes) { if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) return availablePresentMode; }

   // VK_PRESENT_MODE_FIFO_KHR is guaranteed to be available on all devices
   return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D App::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
   // checks if windowing system has a defined window size
   if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return capabilities.currentExtent;

   // else determines a new window size
   int width, height;
   glfwGetFramebufferSize(m_window, &width, &height);

   // returns actual window size
   return {
      std::clamp((uint32_t)width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
      std::clamp((uint32_t)height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
   };
}

void App::createSwapChainImageViews() {
   // makes sure that number of image views is equal to number of images
   m_swapChainImageViews.resize(m_swapChainImages.size());

   // creates a view for each image
   for (uint32_t i = 0; i < m_swapChainImages.size(); i++) {
      m_swapChainImageViews[i] = create2DImageView(m_swapChainImages[i], m_swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
   }
}
void App::createFramebuffers() {
   // makes sure that number of frame buffers is equal to number of image views
   m_swapChainFramebuffers.resize(m_swapChainImageViews.size());

   // create a frame buffer for each image view
   for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
      std::array<VkImageView, 3> attachments = { m_colorImageView, m_depthImageView, m_swapChainImageViews[i] };

      VkFramebufferCreateInfo framebufferInfo = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .renderPass = m_renderPass,
         .attachmentCount = (uint32_t)attachments.size(),
         .pAttachments = attachments.data(),
         .width = m_swapChainExtent.width,
         .height = m_swapChainExtent.height,
         .layers = 1,
      };
      if (vkCreateFramebuffer(m_logicalDevice, &framebufferInfo, DEFAULT_ALLOCATOR, &m_swapChainFramebuffers[i]) != VK_SUCCESS)
         throw std::runtime_error("failed to create framebuffer!");
   }
}


/*** Command Buffer ***/

void App::createCommandPool() {
   // create the command pool
   VkCommandPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = m_queueFamilyIndices.graphicsAndComputeFamily.value()
   };
   if (vkCreateCommandPool(m_logicalDevice, &poolInfo, nullptr, &m_commandBuffersPool) != VK_SUCCESS)
      throw std::runtime_error("failed to create command pool!");
}
void App::createCommandBuffers(std::vector<VkCommandBuffer>& commandBuffer) {
   // make sure the number of command buffers are equal to the number of frames in flight
   commandBuffer.resize(MAX_FRAMES_IN_FLIGHT);

   // allocate the command buffers
   VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = m_commandBuffersPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = (uint32_t)commandBuffer.size(),
   };
   if (vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, commandBuffer.data()) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate command buffers!");
}


/*** Pipelines ***/

void App::createPipelineCache(VkPipelineCache& pipelineCache) {
   // used in initializing a pipeline cache object and storing pre-made cache data
   VkPipelineCacheCreateInfo pipelineCacheInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .initialDataSize = 0,
      .pInitialData = nullptr
   };

   // creates a pipeline cache object that can be used to store and reuse data that is expensive to create
   if (vkCreatePipelineCache(m_logicalDevice, &pipelineCacheInfo, DEFAULT_ALLOCATOR, &pipelineCache) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline cache!");
}

void App::createComputePipeline() {
   // shader stage info structures used for specifying the shader used in the pipeline
   std::vector<uint32_t> compShaderCode = compile("shaders/compute_shader.comp", shaderc_compute_shader);
   VkShaderModule compShaderModule = createShaderModule(compShaderCode);
   VkPipelineShaderStageCreateInfo compShaderStageInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = compShaderModule,
      .pName = "main",
      .pSpecializationInfo = nullptr
   };

   VkPushConstantRange pushConstantRange = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(PushConstantData)
   };

   VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &m_computeDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange
   };

   if (vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutInfo, DEFAULT_ALLOCATOR, &m_computePipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline layout!");

   VkComputePipelineCreateInfo pipelineInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = compShaderStageInfo,
      .layout = m_computePipelineLayout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1
   };

   createPipelineCache(m_computePipelineCache);

   if (vkCreateComputePipelines(m_logicalDevice, m_computePipelineCache, 1, &pipelineInfo, DEFAULT_ALLOCATOR, &m_computePipeline) != VK_SUCCESS)
      throw std::runtime_error("failed to create compute pipeline!");

   vkDestroyShaderModule(m_logicalDevice, compShaderModule, DEFAULT_ALLOCATOR);
}
void App::recreateComputePipeline() {
   vkQueueWaitIdle(m_computeQueue);
   // Destroy the old pipeline
   vkDestroyPipeline(m_logicalDevice, m_computePipeline, DEFAULT_ALLOCATOR);

   // shader stage info structures used for specifying the shader used in the pipeline
   std::vector<uint32_t> compShaderCode = compile("shaders/compute_shader.comp", shaderc_compute_shader);
   VkShaderModule compShaderModule = createShaderModule(compShaderCode);
   VkPipelineShaderStageCreateInfo compShaderStageInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = compShaderModule,
      .pName = "main",
      .pSpecializationInfo = nullptr
   };

   VkPushConstantRange pushConstantRange = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(PushConstantData)
   };

   VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &m_computeDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange
   };

   if (vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutInfo, DEFAULT_ALLOCATOR, &m_computePipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline layout!");

   VkComputePipelineCreateInfo pipelineInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = compShaderStageInfo,
      .layout = m_computePipelineLayout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1
   };

   if (vkCreateComputePipelines(m_logicalDevice, m_computePipelineCache, 1, &pipelineInfo, DEFAULT_ALLOCATOR, &m_computePipeline) != VK_SUCCESS)
      throw std::runtime_error("failed to create compute pipeline!");

   vkDestroyShaderModule(m_logicalDevice, compShaderModule, DEFAULT_ALLOCATOR);
}

void App::createGraphicsPipeline() {
   // shader stages info structures used for specifying the shaders used in the pipeline
   std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

   // vertex shader
   std::vector<uint32_t> vertShaderCode = compile("shaders/vertex_shader.vert", shaderc_vertex_shader);
   VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
   shaderStages[0] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertShaderModule,
      .pName = "main",
      .pSpecializationInfo = nullptr
   };
   // fragment shader
   std::vector<uint32_t> fragShaderCode = compile("shaders/fragment_shader.frag", shaderc_fragment_shader);
   VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
   shaderStages[1] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragShaderModule,
      .pName = "main",
      .pSpecializationInfo = nullptr
   };

   // vertex input info used for specifying how each vertex should be read
   VkVertexInputBindingDescription bindingDescriptions = Vertex::getBindingDescription();
   std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = Vertex::getAttributeDescriptions();
   VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescriptions,
      .vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size(),
      .pVertexAttributeDescriptions = attributeDescriptions.data()
   };

   // input assembly info used for specifying the type of geometry that will be drawn
   VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE
   };

   // viewport info used for specifying the region of the framebuffer that the output will be rendered to
   VkPipelineViewportStateCreateInfo viewportState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr
   };

   // rasterization info used for specifying how the geometry should be rasterized
   VkPipelineRasterizationStateCreateInfo rasterizer = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth = 1.0f
   };

   // multisampling info used for specifying the tequniques uesed in multisampling
   VkPipelineMultisampleStateCreateInfo multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .rasterizationSamples = m_msaaSamples,
      .sampleShadingEnable = VK_TRUE,
      .minSampleShading = 0.1f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE
   };

   // depth and stencil testing info used for specifying how the depth and stencil tests should be performed
   VkPipelineDepthStencilStateCreateInfo depthAndStencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
      .front = {},
      .back = {},
      .minDepthBounds = 0.0f,
      .maxDepthBounds = 1.0f
   };

   // additional color blending info included in the color blending state info
   VkPipelineColorBlendAttachmentState colorBlendAttachment = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
   };

   // color blending info used for specifying how the colors of an image(s) should be blended
   VkPipelineColorBlendStateCreateInfo colorBlending = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
      .blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
   };

   // dynamic state info used for a dynamic viewport and scissor
   std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dynamicState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .dynamicStateCount = (uint32_t)dynamicStates.size(),
      .pDynamicStates = dynamicStates.data()
   };

   // push constant range info used for pushing constant values into the shader
   /*VkPushConstantRange pushConstantRange = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset = 0,
      .size = sizeof(PushConstantData)
   };*/

   // pipeline layout info used to describe the layout of the graphics pipeline
   VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &m_graphicsDescriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
   };

   // creates the graphics pipeline layout object
   if (vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutInfo, DEFAULT_ALLOCATOR, &m_graphicsPipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline layout!");

   // pipeline info used in creation of the graphics pipeline object
   // most previous objects funnels into this object
   VkGraphicsPipelineCreateInfo pipelineInfo = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stageCount = 2,
      .pStages = shaderStages.data(),
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pTessellationState = nullptr,// need to research tessellation stages
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthAndStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = m_graphicsPipelineLayout,
      .renderPass = m_renderPass,
      .subpass = 0,// need to make the pipeline use subpasses
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1
   };

   createPipelineCache(m_graphicsPipelineCache);

   // creates the graphics pipeline object using all the configuration data and shaders
   if (vkCreateGraphicsPipelines(m_logicalDevice, m_graphicsPipelineCache, 1, &pipelineInfo, DEFAULT_ALLOCATOR, &m_graphicsPipeline) != VK_SUCCESS)
      throw std::runtime_error("failed to create graphics pipeline!");

   // clean up the shader code modules
   vkDestroyShaderModule(m_logicalDevice, vertShaderModule, DEFAULT_ALLOCATOR);
   vkDestroyShaderModule(m_logicalDevice, fragShaderModule, DEFAULT_ALLOCATOR);
}


/*** Rendering ***/

void App::recordComputePipelineCmds(VkCommandBuffer commandBuffer) {
   VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = 0,
      .pInheritanceInfo = nullptr
   };

   if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
      throw std::runtime_error("failed to begin recording compute command buffer!");

   vkCmdPushConstants(commandBuffer, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantData), &m_pushConstantData);

   vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);

   vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0, 1, &m_computeDescriptorSets[m_frameInFlightIndex], 0, nullptr);
   vkCmdDispatch(commandBuffer, m_swapChainExtent.width, m_swapChainExtent.height, 1);

   if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
      throw std::runtime_error("failed to record compute command buffer!");
}

void App::createRenderPass() {
   // color attachment (multi-sample pixels)
   VkAttachmentDescription colorAttachment = {
      .flags = 0,
      .format = m_swapChainImageFormat,
      .samples = m_msaaSamples,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
   };

   // resolved color attachment (single sample pixels)
   VkAttachmentDescription resolvedColorAttachment = {
      .flags = 0,
      .format = m_swapChainImageFormat,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
   };

   // depth buffer attachment
   std::vector<VkFormat> formatCandidates = { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT };
   VkAttachmentDescription depthAttachment = {
      .flags = 0,
      .format = findSupportedFormat(formatCandidates, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT),
      .samples = m_msaaSamples,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
   };

   // reference to the color attachment (multi-sample pixels) for current subpass
   VkAttachmentReference colorAttachmentRef = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
   };

   // reference to the resolved color attachment (single sample pixels) for current subpass
   VkAttachmentReference resolvedColorAttachmentRef = {
      .attachment = 2,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
   };

   // reference to the depth attachment for current subpass
   VkAttachmentReference depthAttachmentRef = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
   };

   // subpass used for specifying the color and depth attachments used in current subpass
   VkSubpassDescription subpass = {
      .flags = 0,
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .inputAttachmentCount = 0,
      .pInputAttachments = nullptr,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachmentRef,
      .pResolveAttachments = &resolvedColorAttachmentRef,
      .pDepthStencilAttachment = &depthAttachmentRef,
      .preserveAttachmentCount = 0,
      .pPreserveAttachments = nullptr
   };

   // subpass dependency used for specifying the dependencies between subpasses
   VkSubpassDependency dependency = {
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      .srcAccessMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
   };

   // render pass info used for specifying the attachments and subpasses used in the render pass
   std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, resolvedColorAttachment };
   VkRenderPassCreateInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .attachmentCount = (uint32_t)attachments.size(),
      .pAttachments = attachments.data(),
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &dependency
   };

   // creates a new render pass object
   if (vkCreateRenderPass(m_logicalDevice, &renderPassInfo, DEFAULT_ALLOCATOR, &m_renderPass) != VK_SUCCESS)
      throw std::runtime_error("failed to create render pass!");
}
void App::recordGraphicsPipelineCmds(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
   // begin recording the command to the buffer
   VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      .pInheritanceInfo = nullptr
   };
   if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
      throw std::runtime_error("failed to begin recording command buffer!");

   VkImageMemoryBarrier imageMemoryBarrier = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image = m_rayStorageImage,
    .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
    }
   };

   vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

   // background color
   std::array<VkClearValue, 2> clearValues{};
   clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
   clearValues[1].depthStencil = { 1.0f, 0 };

   // begin the render pass
   VkRenderPassBeginInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .pNext = nullptr,
      .renderPass = m_renderPass,
      .framebuffer = m_swapChainFramebuffers[imageIndex],
      .renderArea = {
         .offset = { 0, 0 },
         .extent = m_swapChainExtent
      },
      .clearValueCount = (uint32_t)clearValues.size(),
      .pClearValues = clearValues.data()
   };
   vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

   // setup for the view port (stretches or strinks the pixels to fit the rectangle)
   VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)m_swapChainExtent.width,
      .height = (float)m_swapChainExtent.height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f
   };
   vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
   // setup for the scissor rectangle (cuts off pixels outside the rectangle)
   VkRect2D scissor = {
      .offset = { 0, 0 },
      .extent = m_swapChainExtent
   };
   vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

   // bind the vertex buffer and index buffers
   VkBuffer vertexBuffers[] = { m_vertexBuffer };
   VkDeviceSize offsets[] = { 0 };
   vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
   vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
   // bind the descriptor sets
   vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineLayout, 0, 1, &m_graphicsDescriptorSets[m_frameInFlightIndex], 0, nullptr);

   // draw the model
   vkCmdDrawIndexed(commandBuffer, (uint32_t)m_indices.size(), 1, 0, 0, 0);
   vkCmdEndRenderPass(commandBuffer);

   // Transition the image back to VK_IMAGE_LAYOUT_GENERAL
   imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
   imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
   imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

   vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

   // stop recording the command to the buffer
   if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
      throw std::runtime_error("failed to record command buffer!");
}

void App::drawFrame() {
   updatePushConstants();
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { updateRayTracerViewUBO(uint32_t(i)); }
   // Compute Stage //

   // waits for the current frame to be finished computing before starting the next frame
   vkWaitForFences(m_logicalDevice, 1, &m_computeInFlightFences[m_frameInFlightIndex], VK_TRUE, UINT64_MAX);
   vkResetFences(m_logicalDevice, 1, &m_computeInFlightFences[m_frameInFlightIndex]);

   vkResetCommandBuffer(m_computeCommandBuffers[m_frameInFlightIndex], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
   recordComputePipelineCmds(m_computeCommandBuffers[m_frameInFlightIndex]);

   // submit info object for command buffer queue submitions
   VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &m_computeCommandBuffers[m_frameInFlightIndex],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &m_computeFinishedSemaphores[m_frameInFlightIndex]
   };

   if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_computeInFlightFences[m_frameInFlightIndex]) != VK_SUCCESS)
      throw std::runtime_error("failed to submit compute command buffer!");

   // Graphics Stage //

   // waits for the current frame to be finished rendering before starting the next frame
   vkWaitForFences(m_logicalDevice, 1, &m_graphicsInFlightFences[m_frameInFlightIndex], VK_TRUE, UINT64_MAX);

   vkResetFences(m_logicalDevice, 1, &m_graphicsInFlightFences[m_frameInFlightIndex]);
   // gets the current frame's swap chain image
   // resizes the swap chain if result is not VK_SUCCESS
   uint32_t imageIndex;
   VkResult result = vkAcquireNextImageKHR(m_logicalDevice, m_swapChain, UINT64_MAX, m_imageAvailableSemaphores[m_frameInFlightIndex], VK_NULL_HANDLE, &imageIndex);
   if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapChain();
      recreateComputeImage();

      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { updateRayTracerViewUBO(uint32_t(i)); }

      m_currentFrame = -1;

      return;
   }
   else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) throw std::runtime_error("failed to acquire swap chain image!");


   vkResetFences(m_logicalDevice, 1, &m_graphicsInFlightFences[m_frameInFlightIndex]);

   // resets the command buffer to be used for rendering
   // creates a draw command from the pipeline and render pass objects to be submited to the command buffer queue
   vkResetCommandBuffer(m_graphicsCommandBuffers[m_frameInFlightIndex], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
   recordGraphicsPipelineCmds(m_graphicsCommandBuffers[m_frameInFlightIndex], imageIndex);

   // submit info object reused for graphics submition
   VkSemaphore waitSemaphores[] = { m_computeFinishedSemaphores[m_frameInFlightIndex], m_imageAvailableSemaphores[m_frameInFlightIndex] };
   VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
   VkSemaphore signalSemaphores[] = { m_graphicsFinishedSemaphores[m_frameInFlightIndex] };
   submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 2,
      .pWaitSemaphores = waitSemaphores,
      .pWaitDstStageMask = waitStages,
      .commandBufferCount = 1,
      .pCommandBuffers = &m_graphicsCommandBuffers[m_frameInFlightIndex],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = signalSemaphores
   };

   // submits a render command to the command buffer queue
   // this is where the main rendering work is done
   if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_graphicsInFlightFences[m_frameInFlightIndex]) != VK_SUCCESS)
      throw std::runtime_error("failed to submit draw command buffer!");

   // present info object for swap chain queue presentation
   VkSwapchainKHR swapChains[] = { m_swapChain };
   VkPresentInfoKHR presentInfo = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = signalSemaphores,
      .swapchainCount = 1,
      .pSwapchains = swapChains,
      .pImageIndices = &imageIndex,
      .pResults = nullptr
   };

   // presents a rendered image to the swap chain queue for displaying to the screen
   // resizes the swap chain if result is not VK_SUCCESS
   result = vkQueuePresentKHR(m_presentQueue, &presentInfo);
   if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
      recreateSwapChain();
      m_framebufferResized = false;
      recreateComputeImage();
      for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { updateRayTracerViewUBO(uint32_t(i)); }

      m_currentFrame = -1;
   }
   else if (result != VK_SUCCESS) throw std::runtime_error("failed to present swap chain image!");

   // swaps the current frame index to the next frame index
   // MAX_FRAMES_IN_FLIGHT:
   // 1 - a to a
   // 2 - a swaps with b, b swaps with a
   // 3 or more - current frame swaps with the next frame; if frame is the last frame, it swaps with the first frame.
   m_frameInFlightIndex = (m_frameInFlightIndex + 1) % MAX_FRAMES_IN_FLIGHT;
   m_currentFrame++;
}


/*** Other ***/

void App::cleanup() {
   cleanupSwapChain();

   // destroy texture image sampler and image buffers
   vkDestroySampler(m_logicalDevice, m_textureSampler, DEFAULT_ALLOCATOR);

   vkDestroyImageView(m_logicalDevice, m_rayStorageImageView, DEFAULT_ALLOCATOR);
   vkDestroyImage(m_logicalDevice, m_rayStorageImage, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, m_rayStorageImageMemory, DEFAULT_ALLOCATOR);

   // destroy graphics pipeline and render pass objects
   vkDestroyPipeline(m_logicalDevice, m_graphicsPipeline, DEFAULT_ALLOCATOR);
   vkDestroyPipelineCache(m_logicalDevice, m_graphicsPipelineCache, DEFAULT_ALLOCATOR);
   vkDestroyPipelineLayout(m_logicalDevice, m_graphicsPipelineLayout, DEFAULT_ALLOCATOR);

   vkDestroyRenderPass(m_logicalDevice, m_renderPass, DEFAULT_ALLOCATOR);
   // free up graphics pipeline uniform buffers memory
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroyBuffer(m_logicalDevice, m_graphicUBOs[i], DEFAULT_ALLOCATOR);
      vkFreeMemory(m_logicalDevice, m_graphicUBOMemory[i], DEFAULT_ALLOCATOR);
   }
   // destroy graphics pipeline descriptor sets
   vkDestroyDescriptorPool(m_logicalDevice, m_pipelineDescriptorSetsPool, DEFAULT_ALLOCATOR);
   vkDestroyDescriptorSetLayout(m_logicalDevice, m_graphicsDescriptorSetLayout, DEFAULT_ALLOCATOR);
   // free up index buffer memory
   vkDestroyBuffer(m_logicalDevice, m_indexBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, m_indexBufferMemory, DEFAULT_ALLOCATOR);
   // free up vertex buffer memory
   vkDestroyBuffer(m_logicalDevice, m_vertexBuffer, DEFAULT_ALLOCATOR);
   vkFreeMemory(m_logicalDevice, m_vertexBufferMemory, DEFAULT_ALLOCATOR);

   // destroy compute pipeline objects
   vkDestroyPipeline(m_logicalDevice, m_computePipeline, DEFAULT_ALLOCATOR);
   vkDestroyPipelineCache(m_logicalDevice, m_computePipelineCache, DEFAULT_ALLOCATOR);
   vkDestroyPipelineLayout(m_logicalDevice, m_computePipelineLayout, DEFAULT_ALLOCATOR);
   // free up compute pipeline uniform buffers memory
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroyBuffer(m_logicalDevice, m_rayViewPortInfoUBOs[i], DEFAULT_ALLOCATOR);
      vkFreeMemory(m_logicalDevice, m_rayViewPortInfoUBOMemory[i], DEFAULT_ALLOCATOR);
   }
   // destroy compute pipeline descriptor sets
   vkDestroyDescriptorSetLayout(m_logicalDevice, m_computeDescriptorSetLayout, DEFAULT_ALLOCATOR);

   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroyBuffer(m_logicalDevice, m_raySpheres[i], DEFAULT_ALLOCATOR);
      vkFreeMemory(m_logicalDevice, m_raySpheresMemory[i], DEFAULT_ALLOCATOR);
   }

   // destroy all syncronization objects
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(m_logicalDevice, m_graphicsFinishedSemaphores[i], DEFAULT_ALLOCATOR);
      vkDestroyFence(m_logicalDevice, m_graphicsInFlightFences[i], DEFAULT_ALLOCATOR);
      vkDestroySemaphore(m_logicalDevice, m_computeFinishedSemaphores[i], DEFAULT_ALLOCATOR);
      vkDestroyFence(m_logicalDevice, m_computeInFlightFences[i], DEFAULT_ALLOCATOR);
      vkDestroySemaphore(m_logicalDevice, m_imageAvailableSemaphores[i], DEFAULT_ALLOCATOR);
   }

   // free up command buffers memory
   vkDestroyCommandPool(m_logicalDevice, m_commandBuffersPool, DEFAULT_ALLOCATOR);

   // destroy Vulkan logical device
   vkDestroyDevice(m_logicalDevice, DEFAULT_ALLOCATOR);

   // destory window surface and Vulkan instance
   vkDestroySurfaceKHR(m_instance, m_surface, DEFAULT_ALLOCATOR);
   GAPIInstance::Destroy();

   // destroy window and GLFW instance
   glfwDestroyWindow(m_window);
   glfwTerminate();
}