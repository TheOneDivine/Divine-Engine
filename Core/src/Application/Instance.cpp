#include "Instance.hpp"
#include "../Platforms/Vulkan/VulkanInstance.hpp"

void* GAPIInstance::Get() {
   switch (GetAPI())
   {
      case 0:
         VulkanInstance instance;
         return instance.Get();
   }

   return nullptr;
}

void GAPIInstance::Destroy() {
   switch (GetAPI())
   {
   case 0:
      VulkanInstance instance;
      instance.Destroy();
   }
}