#pragma once

#include "../../preComp.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../../Application/Instance.hpp"

static const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

// proxy function for creating "VkDebugUtilsMessengerEXT" object
static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
   auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
   if (func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
   else return VK_ERROR_EXTENSION_NOT_PRESENT;
}

// proxy function for destroying "VkDebugUtilsMessengerEXT" object
static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
   auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
   if (func != nullptr) func(instance, debugMessenger, pAllocator);
}

class VulkanInstance : public GAPIInstance
{
public:
   VulkanInstance();

   ~VulkanInstance();

   VulkanInstance(const VulkanInstance&) = delete;
   VulkanInstance& operator=(const VulkanInstance&) = delete;
   
   static VkInstance& Get();

   static void Destroy();

private:
   bool CheckValidationLayerSupport();

   std::vector<const char*> GetRequiredGLFWExtensions();

   static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

   void SetupDebugMessenger();

   static inline VkInstance m_vulkanInstance = nullptr;
   static inline VkDebugUtilsMessengerEXT m_debugMessenger = nullptr;
};