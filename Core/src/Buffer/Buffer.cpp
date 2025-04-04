#include "Buffer.hpp"
#include "../Platforms/Vulkan/VulkanBuffer.hpp"

Ref<VertexBuffer> VertexBuffer::create(size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanVertexBuffer>(size);
   }

   return nullptr;
}

Ref<VertexBuffer> VertexBuffer::create(void* vertices, size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanVertexBuffer>(vertices, size);
   }

   return nullptr;
}

Ref<IndexBuffer> IndexBuffer::create(uint32_t* indices, size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanIndexBuffer>(indices, size);
   }

   return nullptr;
}

Ref<UniformBuffer> UniformBuffer::create(size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanUniformBuffer>(size);
   }

   return nullptr;
}

Ref<UniformBuffer> UniformBuffer::create(void* data, size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanUniformBuffer>(data, size);
   }

   return nullptr;
}

Ref<ShaderStorageBuffer> ShaderStorageBuffer::create(size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanShaderStorageBuffer>(size);
   }

   return nullptr;
}

Ref<ShaderStorageBuffer> ShaderStorageBuffer::create(void* data, size_t size) {
   switch (getAPI())
   {
      case 0: return createRef<VulkanShaderStorageBuffer>(data, size);
   }

   return nullptr;
}