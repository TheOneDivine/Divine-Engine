#include "VulkanInstance.hpp"

VulkanInstance::VulkanInstance() {
   if (m_vulkanInstance) return;

   // checks to validation layer support
   if (_DEBUGGING_ENABLED && !CheckValidationLayerSupport())
      throw std::runtime_error("validation layers requested, but not available!");

   // application info
   VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = TITLE,
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(0, 0, 0),
      .apiVersion = VK_API_VERSION_1_4
   };

   // Vulkan instance creat info
   std::vector<const char*> extensions = GetRequiredGLFWExtensions();
   VkInstanceCreateInfo createInfo{};
   if (_DEBUGGING_ENABLED) {
      // set up validation layers
      VkDebugUtilsMessengerCreateInfoEXT* debugCreateInfo{};
      createInfo = {
         .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
         .pNext = debugCreateInfo,
         .flags = 0,
         .pApplicationInfo = &appInfo,
         .enabledLayerCount = (uint32_t)validationLayers.size(),
         .ppEnabledLayerNames = validationLayers.data(),
         .enabledExtensionCount = (uint32_t)extensions.size(),
         .ppEnabledExtensionNames = extensions.data()
      };
   }
   else {
      // disable validation layers
      createInfo = {
         .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .pApplicationInfo = &appInfo,
         .enabledLayerCount = 0,
         .ppEnabledLayerNames = nullptr,
         .enabledExtensionCount = (uint32_t)extensions.size(),
         .ppEnabledExtensionNames = extensions.data()
      };
   }

   // create Vulkan instance
   if (vkCreateInstance(&createInfo, DEFAULT_ALLOCATOR, &Get()) != VK_SUCCESS)
      throw std::runtime_error("failed to create Vulkan instance!");
}

VulkanInstance::~VulkanInstance() {}

VkInstance& VulkanInstance::Get() { return m_vulkanInstance; }

void VulkanInstance::Destroy() {
   if (m_vulkanInstance) {
      // stop Vulkan debug messenger
      if (_DEBUGGING_ENABLED)
         DestroyDebugUtilsMessengerEXT(m_vulkanInstance, m_debugMessenger, DEFAULT_ALLOCATOR);

      vkDestroyInstance(m_vulkanInstance, DEFAULT_ALLOCATOR);
      m_vulkanInstance = nullptr;
   }
}

bool VulkanInstance::CheckValidationLayerSupport() {
   // get number of supported validation layers
   uint32_t layerCount;
   vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
   // list supported validation layers
   std::vector<VkLayerProperties> availableLayers(layerCount);
   vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

   // check for required validation layers
   for (const char* layerName : validationLayers) {
      bool layerFound = false;

      // check for validation layer properties
      for (const auto& layerProperties : availableLayers) {
         if (strcmp(layerName, layerProperties.layerName) == 0) { layerFound = true; break; }
      }
      if (!layerFound) return false;
   }

   // if all requirments are met return true
   return true;
}

std::vector<const char*> VulkanInstance::GetRequiredGLFWExtensions() {
   // gets Vulkan extensions for GLFW compatiblity
   uint32_t glfwExtensionCount = 0;
   const char** glfwExtensions;
   glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

   // handle to extensions
   std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

   // enables validation layer extension
   if (_DEBUGGING_ENABLED) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
   // else returns just the required extinstions
   return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanInstance::DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
   // outputs validation layer error information
   std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
   return VK_FALSE;
}

void VulkanInstance::SetupDebugMessenger() {
   // does nothing if not in debug mode
   if (!_DEBUGGING_ENABLED) return;

   // populates and creates the Vulkan debug messenger
   VkDebugUtilsMessengerCreateInfoEXT createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .pNext = nullptr,
      .flags = 0,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = DebugCallback,
      .pUserData = nullptr,
   };
   if (CreateDebugUtilsMessengerEXT(m_vulkanInstance, &createInfo, DEFAULT_ALLOCATOR, &m_debugMessenger) != VK_SUCCESS)
      throw std::runtime_error("failed to set up debug messenger!");

}