#pragma once

#include <vulkan/vulkan.h>

#include "../../Buffer/Buffer.hpp"

class VulkanBuffer
{
public:
   

protected:
   void createBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
   void copyBuffer(VkBuffer& dstBuffer, VkDeviceSize size, VkCommandPool commandPool, VkQueue commandQueue);
   uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

   VkDevice m_logicalDevice;
   VkPhysicalDevice m_physicalDevice;
   VkBuffer m_buffer;
   VkDeviceMemory m_bufferMemory;

private:

};


//********** Vertex Buffer **********//

class VulkanVertexBuffer : public VulkanBuffer, public VertexBuffer
{
public:
   VulkanVertexBuffer(size_t size);
   VulkanVertexBuffer(void* vertices, size_t size);
   ~VulkanVertexBuffer();

   void bind() const override;
   void unbind() const override;

   void setData(const void* data, size_t size) override;
   const BufferLayout& getLayout() const override;
   void setLayout(const BufferLayout& layout) override;

private:
   BufferLayout m_layout;
};


//********** Index Buffer **********//

class VulkanIndexBuffer : public VulkanBuffer, public IndexBuffer
{
public:
   VulkanIndexBuffer(uint32_t* indices, size_t count);
   ~VulkanIndexBuffer();

   void bind() const override;
   void unbind() const override;

   size_t getCount() const override;

private:
    size_t m_count;
};


//********** Uniform Buffer **********//

class VulkanUniformBuffer : public VulkanBuffer, public UniformBuffer
{
public:
   VulkanUniformBuffer(size_t size);
   VulkanUniformBuffer(void* data, size_t size);
   ~VulkanUniformBuffer();

   void bind() const override;
   void unbind() const override;

   void setData(const void* data, size_t size) override;
   const BufferLayout& getLayout() const override;
   void setLayout(const BufferLayout& layout) override;

private:
   BufferLayout m_layout;

};


//********** Shader Storage Buffer **********//

class VulkanShaderStorageBuffer : public VulkanBuffer, public ShaderStorageBuffer
{
public:
   VulkanShaderStorageBuffer(size_t size);
   VulkanShaderStorageBuffer(void* data, size_t size);
   ~VulkanShaderStorageBuffer();

   void bind() const override;
   void unbind() const override;

   void setData(const void* data, size_t size) override;
   const BufferLayout& getLayout() const override;
   void setLayout(const BufferLayout& layout) override;

private:
   BufferLayout m_layout;
};