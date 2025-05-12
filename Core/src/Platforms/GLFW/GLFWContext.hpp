#pragma once

#include "../../preComp.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace GLFW
{
   // GLFW window constext for Vulkan
   class GLFWContext
   {
   public:
      GLFWContext(VkInstance instance);
      void Destroy();

      uint32_t InitWindow(int width, int height, const char* title, GLFWmonitor* monitor, GLFWwindow* share);
      void DestroyWindow(uint32_t ID);
      void DestroyWindow(GLFWwindow* targetWindow);

      uint32_t CreateSurface(uint32_t windowID);
      void DestroySurface(uint32_t surfaceID);
      void DestroySurface(VkSurfaceKHR surface);

   private:
      static std::vector<const char*> getRequiredExtensions();

      VkInstance m_vulkanInstance;

      std::list<Ref<GLFWwindow>> m_windows;
      std::list<Ref<VkSurfaceKHR>> m_surfaces;
   };
}