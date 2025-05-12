#include "Buffer.hpp"
#include "../Platforms/Vulkan/VulkanBuffer.hpp"

Ref<VertexBuffer> VertexBuffer::create(size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanVertexBuffer>(size);
   }

   return nullptr;
}

Ref<VertexBuffer> VertexBuffer::create(void* vertices, size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanVertexBuffer>(vertices, size);
   }

   return nullptr;
}

Ref<IndexBuffer> IndexBuffer::create(uint32_t* indices, size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanIndexBuffer>(indices, size);
   }

   return nullptr;
}

Ref<UniformBuffer> UniformBuffer::create(size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanUniformBuffer>(size);
   }

   return nullptr;
}

Ref<UniformBuffer> UniformBuffer::create(void* data, size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanUniformBuffer>(data, size);
   }

   return nullptr;
}

Ref<ShaderStorageBuffer> ShaderStorageBuffer::create(size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanShaderStorageBuffer>(size);
   }

   return nullptr;
}

Ref<ShaderStorageBuffer> ShaderStorageBuffer::create(void* data, size_t size) {
   switch (GetAPI())
   {
      case 0: return CreateRef<VulkanShaderStorageBuffer>(data, size);
   }

   return nullptr;
}