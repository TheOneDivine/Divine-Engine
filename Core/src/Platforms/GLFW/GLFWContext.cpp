#include "GLFWContext.hpp"

namespace GLFW
{
   GLFWContext::GLFWContext(VkInstance instance) : m_vulkanInstance(instance) {
      // creates GLFW instance
      
      //_ASSERT(glfwInit() != GLFW_TRUE, "Failed to initialize GLFW!");

      glfwInit();

      // enables/disables GLFW features
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
   }

   uint32_t GLFWContext::InitWindow(int width, int height, const char* title, GLFWmonitor* monitor, GLFWwindow* share) {
      // creates window instance and initiates client callbacks
      GLFWwindow* window = glfwCreateWindow(width, height, title, monitor, share);
      //_ASSERT(!window, "Failed to create window!");

      m_windows.push_back(createRef<GLFWwindow>(window));

      return (uint32_t)m_windows.size();
   }

   void GLFWContext::DestroyWindow(uint32_t ID) {
      // destroy surface
      auto surface = std::find(m_surfaces.begin(), m_surfaces.end(), ID);
      if (surface != m_surfaces.end()) DestroySurface(*surface->get());

      // destroy window
      auto window = std::find(m_windows.begin(), m_windows.end(), ID);
      if (window != m_windows.end()) glfwDestroyWindow(window->get());

      // remove pointer from list
      m_windows.erase(window);
   }
   void GLFWContext::DestroyWindow(GLFWwindow* targetWindow) {
      int ID = 0;
      for (auto it = m_windows.begin(); it != m_windows.end(); ++it, ++ID) {
         if ((*it).get() == targetWindow) { DestroyWindow(ID); }
      }
   }

   uint32_t GLFWContext::CreateSurface(uint32_t windowID) {
      auto window = std::find(m_windows.begin(), m_windows.end(), windowID);
      if (window != m_windows.end()) {

         // using GLFW creates a Vulkan window surface
         VkSurfaceKHR surface;
         glfwCreateWindowSurface(m_vulkanInstance, window->get(), DEFAULT_ALLOCATOR, &surface);
         //_ASSERT(glfwCreateWindowSurface(m_vulkanInstance, window->get(), DEFAULT_ALLOCATOR, &surface) != VK_SUCCESS,
         //   "Failed to create window surface");

         m_surfaces.push_back(createRef<VkSurfaceKHR>(surface));

         return (uint32_t)m_surfaces.size();
      }

      //_ASSERT(false, "Window does not exist!");
   }
   void GLFWContext::DestroySurface(uint32_t surfaceID) {
      // destroy surface
      auto surface = std::find(m_surfaces.begin(), m_surfaces.end(), surfaceID);
      if (surface != m_surfaces.end()) {
         vkDestroySurfaceKHR(m_vulkanInstance, *surface->get(), DEFAULT_ALLOCATOR);
         m_surfaces.erase(surface);
         return;
      }

      //_ASSERT(false, "Failed to delete surface!");
   }
   void GLFWContext::DestroySurface(VkSurfaceKHR targetSurface) {
      int ID = 0;
      for (auto it = m_surfaces.begin(); it != m_surfaces.end(); ++it, ++ID) {
         if (*it->get() == targetSurface) DestroySurface(ID);
      }
   }

   void GLFWContext::Destroy() {
      for (auto& windows : m_windows) {
         if (windows) DestroyWindow(windows.get());
      }

      glfwTerminate();
   }

   std::vector<const char*> GLFWContext::getRequiredExtensions() {
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
}