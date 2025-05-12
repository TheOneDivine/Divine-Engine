#pragma once

#include "pch.hpp"

// External libraries //
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <shaderc/shaderc.hpp>

// Standard Library //
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>
#include <random>
#include <thread>

// need to make memory allocator (HUGE performance gains)
//#define DEFAULT_ALLOCATOR nullptr

/*
GLSL alignment requirements using alignas():
   int, uint, float = 4
   double = 8
   vec2, dvec2 = 8
   vec3, vec4, dvec3, dvec4 = 16
   mat2, mat3, mat4, dmat2, dmat3, dmat4 = 16
   array = number of elements times largest element type alignment (e.g. array of int[12] = 4 * 12 = 48 bytes)
   custom struct = each member aligns according to its type, and the struct aligns to a multiple of its largest member's alignment (e.g. if largest member is 16 bytes, the struct aligns to 16 bytes).
*/

// vertex binding descriptions
struct Vertex
{
   alignas(16) glm::vec3 pos;
   alignas(16) glm::vec3 color;
   alignas(16) glm::vec2 texCoord;

   static VkVertexInputBindingDescription getBindingDescription() {
      VkVertexInputBindingDescription bindingDescription{};
      bindingDescription.binding = 0;
      bindingDescription.stride = sizeof(Vertex);
      bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      return bindingDescription;
   }
  
   static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
      std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

      attributeDescriptions[0].binding = 0;
      attributeDescriptions[0].location = 0;
      attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
      attributeDescriptions[0].offset = offsetof(Vertex, pos);

      attributeDescriptions[1].binding = 0;
      attributeDescriptions[1].location = 1;
      attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
      attributeDescriptions[1].offset = offsetof(Vertex, color);

      attributeDescriptions[2].binding = 0;
      attributeDescriptions[2].location = 2;
      attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
      attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

      return attributeDescriptions;
   }

   bool operator==(const Vertex& other) const {
      return pos == other.pos && color == other.color && texCoord == other.texCoord;
   }
};
// vertex hash object
namespace std {
   template<> struct hash<Vertex> {
      size_t operator()(Vertex const& vertex) const {
         return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec2>()(vertex.texCoord) << 1)) >> 1);
      }
   };
}


// Push Constant Data
struct PushConstantData
{
   alignas(8) double deltaTime = 0.0f;
   alignas(4) int currentFrame = 1;
};

// compute shaders UBO for viewport data
typedef struct ComputeViewPortUniformBufferObject
{
   alignas(8)  glm::vec2 winExtent = { 1, 1 };
   alignas(16) glm::vec3 firstPixel = { 0.0f, 0.0f, 0.0f };
   alignas(16) glm::vec3 pixelDeltaU = { 0.0f, 0.0f, 0.0f };
   alignas(16) glm::vec3 pixelDeltaV = { 0.0f, 0.0f, 0.0f };
   alignas(16) glm::vec3 defocusDiskU = { 0.0f, 0.0f, 0.0f };
   alignas(16) glm::vec3 defocusDiskV = { 0.0f, 0.0f, 0.0f };
   alignas(4) float defocusAngle = 0.0f;
   alignas(16) glm::vec3 cameraCenter = { 0.0f, 0.0f, 0.0f };
   alignas(4) bool isRendering = false;
} RayViewData;

// ray tracer camera
typedef struct Camera
{
   glm::vec3 center = { 13.0f, 2.0f, 3.0f }; // center of camera
   glm::vec3 lookat = { 0.0f, 0.0f, 0.0f };                         // direction camera is looking
   glm::vec3 vup = { 0.0f, 1.0f, 0.0f };     // Camera-relative "up" direction
   float fov = glm::radians(20.0f);          // cameras field of view
   float defocusAngle = 0.6f;                // 
   float defocusDistance = 10.0f;            // 

} Camera;

typedef enum Ray_Material_Type
{
   LAMBERTIAN = 0,       // rays are scattered randomly creating a matte look
   METAL = 1,            // rays are scattered more evenly (in some direction) to create more reflective surfaces (mirrors and metals)
   DIELECTRIC = 2,       // rays pass through objects and can reflect or refract (glass and water)
   MAT_DEFAULT = LAMBERTIAN, // default material set when creating a new meterial
   MAT_MAX_ENUM = 0x7FFF // max integer value currently used by ray shader
} MaterialType;

typedef enum Ray_Object_Type
{
   SPHERE = 0,
   QUADRILATERAL = 1,
   OBJ_DEFAULT = SPHERE, // default object set when creating a new object
   OBJ_MAX_ENUM = 0x7FFF // max integer value currently used by ray shader
} ObjectType;

// graphical shaders UBO
typedef struct GraphicalUniformBufferObject
{
   alignas(16) glm::mat4 model;
   alignas(16) glm::mat4 view;
   alignas(16) glm::mat4 proj;
} GraphicUBO;

// Required validation layers and extensions for vulkan

//static const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
//static const std::vector<const char*> deviceExtensions = { "VK_KHR_swapchain" };

// Global Variables //

//static const char* TITLE = "Hello Triangle"; // app title
//static const uint32_t WIDTH = 800;          // app width (in pixels)
//static const uint32_t HEIGHT = 400;         // app height (in pixels)

static const size_t MAX_OBJECT_COUNT = 1000000;                                    // max number of objects in the scene
static const size_t MAX_FRAMES_IN_FLIGHT = 1;                                      // defines the max frames allowed in the render stage at once
static const size_t NUMBER_OF_LOGICAL_CORES = std::thread::hardware_concurrency(); // number of CPU logical proccessors
static const size_t MAX_THREAD_COUNT = NUMBER_OF_LOGICAL_CORES * 10;               // max number of threads
static const size_t NUMBER_OF_COMMAND_BUFFERS = 0;                                 // number of command buffers

static const uint32_t PARTICLE_COUNT = 256000 / 2;

const std::string MODEL_PATH = "models/viking_room.obj";            // path to model file
const std::string TEXTURE_PATH = "images/blank.png"; // path to texture file

// structure for storage of available queue families
struct QueueFamilyIndices
{
   std::optional<uint32_t> graphicsAndComputeFamily;
   std::optional<uint32_t> presentFamily;

   bool isComplete() {
     return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
   }
};

// structure for storage of window surface properties
struct SwapChainSupportDetails
{
   VkSurfaceCapabilitiesKHR capabilities{ 0 };
   std::vector<VkSurfaceFormatKHR> formats;
   std::vector<VkPresentModeKHR> presentModes;
};

// reads a file into a buffer
static inline std::vector<char> readFile(const std::string& filename) {
   std::ifstream file(filename, std::ios::ate | std::ios::binary);

   if (!file.is_open())
      throw std::runtime_error("failed to open file!");

   size_t fileSize = (size_t)file.tellg();
   std::vector<char> buffer(fileSize);

   file.seekg(0);
   file.read(buffer.data(), fileSize);

   file.close();

   return buffer;
}


static inline std::vector<char> compile(const std::string& filename, shaderc_shader_kind type) {
   std::string shaderSource = readFile(filename).data();

   // Create Shaderc compiler and options
   shaderc::Compiler compiler;
   shaderc::CompileOptions options;

#if defined(DEBUG)
   options.SetOptimizationLevel(shaderc_optimization_level_zero);
   options.SetGenerateDebugInfo();
   options.AddMacroDefinition("DEBUG", "1");

#elif defined(RELEASE)
   options.SetOptimizationLevel(shaderc_optimization_level_zero);
   options.SetGenerateDebugInfo();
   options.AddMacroDefinition("RELEASE", "1");

#else
   options.SetOptimizationLevel(shaderc_optimization_level_performance);

#endif

   // Compile GLSL to SPIR-V
   shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(shaderSource, type, filename.c_str(), options);

   // Check compilation result
   if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
      const std::string errorMessage = "Shader compilation failed: " + result.GetErrorMessage();
      std::runtime_error(errorMessage.c_str());
   }

   return readFile(filename);
}

// random number generator
static float randomFloat() {
   return float(std::rand()) / (RAND_MAX + 1.0f);
}
static float randomFloat(float min, float max) {
   return min + (max - min) * randomFloat();
}

#include "aabb.hpp"
#include "Buffer/Buffer.hpp"
#include "Scene/Entity.hpp"

// Main App class (mostly vulkan boilerplate)
class App
{
public:

   // App initialization
   App() {
      initGLFW();
      m_startTime = std::chrono::high_resolution_clock::now();
      initVulkan();
   }

   // Main Game Loop
   void run() {
      while (!glfwWindowShouldClose(m_window)) {
         glfwPollEvents();
         std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
         m_lastFrameTime = std::chrono::duration<double, std::chrono::seconds::period>(currentTime - m_startTime).count() * 1000;
         m_startTime = currentTime;
         drawFrame();
      }
      vkDeviceWaitIdle(m_logicalDevice);
   }

   // App cleanup
   ~App() {
      cleanup();
   }

private:

   /**********  Vulkan Setup  **********/

   // initiates Vulkan objects and prepares the application for rendering
   void initVulkan();

   /*** GLFW Window ***/

   // initiates GLFW and creates a window
   void initGLFW();
   static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
   static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
   // GLFW callback for the resizing of the window frame buffer (client area)
   static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
   // creates Vulkan surface for the specified window
   void createSurface();
   // checks for all Vulkan extenstions required by GLFW
   std::vector<const char*> getRequiredExtensions();


   /**********  Physical and Logical Device  **********/

   // sets the most optimal physical device for the application
   void pickPhysicalDevice();
   // initiates a VkDeviceCreateInfo object and creates a logical device
   void createLogicalDevice();
   // determines if a physical device features are suitable for application
   bool isDeviceSuitable(VkPhysicalDevice device);
   // returns the maximum SampleCount value for MSAA
   VkSampleCountFlagBits getMaxUsableSampleCount();
   // returns all Vulkan extensions supported by current device
   bool checkDeviceExtensionSupport(VkPhysicalDevice device);
   // returns all queue families supported by the physical device
   QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
   // initiates VkSemaphoreCreateInfo and VkFenceCreateInfo objects and uses them to syncronize CPU and GPU operations
   void createSyncObjects();


   /**********  Descriptor Buffer  **********/

   // initiates a VkMemoryAllocateInfo object, allocates memory using that object and binds it to a buffer
   void createDescriptorBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
   // creates a command for copying source buffer data to a destination buffer
   void copyDescriptorBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkQueue commandQueue);
   // initiates a VkShaderModuleCreateInfo object and wraps shader code in it
   VkShaderModule createShaderModule(const std::vector<char>& code);


   /********** Pipeline Descriptors **********/

   // pushes new data to the global shader constant data
   void updatePushConstants();

   void loadScene();

   // maps a host (CPU) compute shader storage buffer object to a Vulkan buffer (GPU)
   void createRayTracerSSBOs();
   // maps host (CPU) compute uniform buffer object to a Vulkan buffer (GPU)
   void createRayTracerViewUBO();
   // updates the compute pipelines uniform buffer object
   void updateRayTracerViewUBO(uint32_t currentImage);

   // maps a host (CPU) vertex buffer object to a Vulkan buffer (GPU)
   void createVertexBuffer();
   // maps a host (CPU) index buffer object to a Vulkan buffer (GPU)
   void createIndexBuffer();
   // maps host (CPU) graphics pipeline uniform buffer object to a Vulkan buffer (GPU)
   void createGraphicsUBO();
   // updates the graphics pipelines uniform buffer object
   void updateGraphicUBO(uint32_t currentImage);


   /**********  Command Buffer  **********/

   // initiates a VkCommandPoolCreateInfo object and creates a command pool
   void createCommandPool();
   // initiates a VkCommandBufferAllocateInfo object and creates a Vulkan command buffer
   void createCommandBuffers(std::vector<VkCommandBuffer>& commandBuffer);


   /**********  Image Buffers  **********/

   // creates a VkMemoryAllocateInfo object and allocates memory for a buffer
   void create2DImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
   // initates a VkImageViewCreateInfo object and creates an image view
   VkImageView create2DImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
   // returns the most optimal physical device memory type for use with image buffers
   uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

   // begins recording a single time use Vulkan command buffer
   VkCommandBuffer beginSingleTimeCommands();
   // stops recording a single time use Vulkan command buffer
   void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue commandQueue);
   // creates a command for transitioning a Vulkan image layout to a proper format for use with texture buffers
   void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, VkQueue commandQueue);
   // copies CPU-accessable buffer of image data to a compatible GPU-optimized image buffer 
   void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, VkQueue commandQueue);

   // creates a command for generating mipmap images from an image
   void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, VkQueue commandQueue);

   // returns the most optimal stencil format for use with depth buffers
   VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);


   /**********  Pipeline Images  **********/

   // creates image buffer resources for storage of input and output images with shaders
   void createComputeImage();

   void recreateComputeImage();

   // creates image buffer resources to store an image (texture) for use with a shader
   void createTextureImage();
   // initiates a VkSamplerCreateInfo object and creates a texture sampler
   void createTextureSampler();
   // creates color attachment resources for the swapchain to temporarily store color data
   void createColorResources();
   // creates depth buffer resources for depth testing and image stencils
   void createDepthResources();


   /**********  Descriptor Sets  **********/

   // initiates a VkDescriptorPoolCreateInfo object and creates a descriptor pool for storing resource descriptor sets
   void createPipelineResourceDescriptorSetsPool();

   // creates a layout definition for a set of descriptors for describing resources that are shared between the compute pipeline and compute shader
   void createComputeResourceDescriptorSetLayout();
   // creates a set of descriptors for describing resources that are shared between the compute pipeline and compute shaders
   void allocateComputeResourceDescriptorSets();

   // creates a layout definition for a set of descriptors for describing resources that are shared between the graphics pipeline and graphics shaders
   void createGraphicsResourceDescriptorSetLayout();
   // creates a set of descriptors for describing resources that are shared between the graphics pipeline and graphics shaders
   void allocateGraphicsResourceDescriptorSets();


   /**********  Pipelines  **********/

   // initiates a VkPipelineCacheCreateInfo and creates a pipeline cache object
   void createPipelineCache(VkPipelineCache& pipelineCache);

   // initiates a VkGraphicsPipelineCreateInfo object for describing graphic shaders and resources relations
   void createGraphicsPipeline();

   // initiates a VkGraphicsPipelineCreateInfo object for describing computation shaders and resources relations
   void createComputePipeline();

   void recreateComputePipeline();


   /**********  Rendering  **********/

   // records all computation related commands into a command buffer (compute command buffer)
   void recordComputePipelineCmds(VkCommandBuffer commandBuffer);

   // initiates a VkRenderPassCreateInfo object for describing how to render data
   void createRenderPass();
   // records all graphics related commands into a command buffer (graphics command buffer)
   void recordGraphicsPipelineCmds(VkCommandBuffer commandBuffer, uint32_t imageIndex);

   // renders current frame(s) and submits them to the frame buffer(s)
   void drawFrame();


   /**********  Swap Chain  **********/

   // initates a VkSwapchainCreateInfoKHR object and creates a swapchain
   void createSwapChain();
   // deletes current swapchain and other attachments then creates new ones
   void recreateSwapChain();
   // destroys all Vulkan objects currently involved in the swapchain(s)
   void cleanupSwapChain();
   
   // intitiates a SwapChainSupportDetails object with supported swapchain details
   SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
   // returns the most optimal swapchain surface format (color format specifics) supported by the physical device
   VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
   // returns the most optimal swapchain present mode (vertical sync) supported by the physical device
   VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
   // returns the most optimal swapchain dimensions (width and height) supported by the physical device
   VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

   // creates swapchain image view(s)
   void createSwapChainImageViews();
   // initates a VkFramebufferCreateInfo object(s) and creates a framebuffer(s)
   void createFramebuffers();


   /*********  Other Functions  *********/

   // loads a 3d model object into compatible vertex and index buffers
   void loadModel();

   // loads the screen space quad for outputing renders to
   void loadQuad();

   // frees up all allocated memory made by the app instance
   void cleanup();


   /********* App Variables *********/


   /*** Window and Instance ***/

   GLFWwindow* m_window;
   VkInstance m_instance;
   VkDebugUtilsMessengerEXT m_debugMessenger;
   VkSurfaceKHR m_surface;


   /*** Physical and Logical Device ***/

   VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
   VkPhysicalDeviceProperties m_physicalDeviceProperties;
   VkDevice m_logicalDevice;
   QueueFamilyIndices m_queueFamilyIndices;
   VkQueue m_graphicsQueue;
   VkQueue m_computeQueue;
   VkQueue m_presentQueue;


   /*** Swap Chain ***/

   VkSwapchainKHR m_swapChain;
   std::vector<VkImage> m_swapChainImages;
   VkFormat m_swapChainImageFormat;
   VkExtent2D m_swapChainExtent;
   std::vector<VkImageView> m_swapChainImageViews;
   std::vector<VkFramebuffer> m_swapChainFramebuffers;


   /*** Shader Image Resources ***/

   VkSampler m_textureSampler;

   VkImage m_rayStorageImage;
   VkImageView m_rayStorageImageView;
   VkDeviceMemory m_rayStorageImageMemory;

   /*** Pipeline Buffer Descriptors ***/

   VkDescriptorPool m_pipelineDescriptorSetsPool;

   VkDescriptorSetLayout m_computeDescriptorSetLayout;
   std::vector<VkDescriptorSet> m_computeDescriptorSets;

   VkDescriptorSetLayout m_graphicsDescriptorSetLayout;
   std::vector<VkDescriptorSet> m_graphicsDescriptorSets;


   /*** Shader Pipeline ***/

   VkPipelineLayout m_computePipelineLayout;
   VkPipelineCache m_computePipelineCache;
   VkPipeline m_computePipeline;

   VkPipelineLayout m_graphicsPipelineLayout;
   VkPipelineCache m_graphicsPipelineCache;
   VkPipeline m_graphicsPipeline;

   VkRenderPass m_renderPass;


   /*** Command Buffers ***/

   VkCommandPool m_commandBuffersPool;

   std::vector<VkCommandBuffer> m_graphicsCommandBuffers;

   std::vector<VkCommandBuffer> m_computeCommandBuffers;


   /*** Depth and Color Resources ***/

   VkImage m_colorImage;
   VkImageView m_colorImageView;
   VkDeviceMemory m_colorImageMemory;

   VkImage m_depthImage;
   VkImageView m_depthImageView;
   VkDeviceMemory m_depthImageMemory;


   /*** Synchronization Objects ***/

   std::vector<VkSemaphore> m_imageAvailableSemaphores;
   std::vector<VkSemaphore> m_graphicsFinishedSemaphores;
   std::vector<VkFence> m_graphicsInFlightFences;

   std::vector<VkSemaphore> m_computeFinishedSemaphores;
   std::vector<VkFence> m_computeInFlightFences;


   /*** Framebuffer ***/

   bool m_framebufferResized = false;
   uint32_t m_frameInFlightIndex = 0;
   double m_lastFrameTime = 0.0f;
   std::chrono::steady_clock::time_point m_startTime;
   int m_currentFrame = 1;


   /*** Compute Shader Object Buffers ***/
   Ref<Scene> m_scene;


   /*** Shader Buffers ***/

   PushConstantData m_pushConstantData{};
   
   bool m_isRendering = false;

   std::vector<VkBuffer> m_rayViewPortInfoUBOs;
   std::vector<void*> m_rayViewPortInfoMappedUBOs;
   std::vector<VkDeviceMemory> m_rayViewPortInfoUBOMemory;

   std::vector<VkBuffer> m_raySpheres;
   std::vector<VkDeviceMemory> m_raySpheresMemory;

   std::vector<VkBuffer> m_rayMaterials;
   std::vector<VkDeviceMemory> m_rayMaterialsMemory;

   std::vector<VkBuffer> m_graphicUBOs;
   std::vector<void*> m_graphicMappedUBOs;
   std::vector<VkDeviceMemory> m_graphicUBOMemory;

   VkBuffer m_vertexBuffer;
   std::vector<Vertex> m_vertices;
   VkDeviceMemory m_vertexBufferMemory;

   std::vector<uint32_t> m_indices;
   VkBuffer m_indexBuffer;
   VkDeviceMemory m_indexBufferMemory;


   /*** MSAA and Mip Levels ***/

   VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;
   uint32_t m_mipLevels;


   /*** User Input ***/
   
   Camera m_rayCamera;
   bool m_firstMouse = true;
   glm::vec2 m_firstMousePos = { 0.0f, 0.0f };
};