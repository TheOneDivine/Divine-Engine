#include "VulkanBuffer.hpp"

//********** Vulkan Buffer **********//

void VulkanBuffer::createBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
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
      if (vkCreateBuffer(m_logicalDevice, &bufferInfo, nullptr, &m_buffer) != VK_SUCCESS)
         throw std::runtime_error("failed to create buffer!");

      // get buffer memory requirements
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(m_logicalDevice, m_buffer, &memRequirements);

      // allocate memory for buffer
      VkMemoryAllocateInfo allocInfo = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .pNext = nullptr,
         .allocationSize = memRequirements.size,
         .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
      };
      if (vkAllocateMemory(m_logicalDevice, &allocInfo, nullptr, &m_bufferMemory) != VK_SUCCESS)
         throw std::runtime_error("failed to allocate buffer memory!");

      // bind memory to buffer
      vkBindBufferMemory(m_logicalDevice, m_buffer, m_bufferMemory, 0);
   }
void VulkanBuffer::copyBuffer(VkBuffer& dstBuffer, VkDeviceSize size, VkCommandPool commandPool, VkQueue commandQueue) {
   // command buffer memory allocation info
   VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = commandPool,
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
   vkCmdCopyBuffer(commandBuffer, m_buffer, dstBuffer, 1, &copyRegion);
   vkEndCommandBuffer(commandBuffer);

   // create a fence to wait for the command buffer to complete
   VkFenceCreateInfo fenceInfo = {
       .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
       .pNext = nullptr,
       .flags = 0
   };
   VkFence fence;
   vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &fence);

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
   vkQueueSubmit(commandQueue, 1, &submitInfo, fence);

   // wait for the fence to signal that command buffer has finished executing
   vkWaitForFences(m_logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

   // free command buffer
   vkFreeCommandBuffers(m_logicalDevice, commandPool, 1, &commandBuffer);
   }
uint32_t VulkanBuffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
   VkPhysicalDeviceMemoryProperties memProperties;
   vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

   for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) { return i; }
   }

   throw std::runtime_error("failed to find suitable memory type!");
   }


//********** Vertex Buffer **********//

VulkanVertexBuffer::VulkanVertexBuffer(size_t size) {
   createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}
VulkanVertexBuffer::VulkanVertexBuffer(void* vertices, size_t size) {
   createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   setData(vertices, size);
}
VulkanVertexBuffer::~VulkanVertexBuffer() {
   if (m_buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_logicalDevice, m_buffer, nullptr);
   if (m_bufferMemory != VK_NULL_HANDLE) vkFreeMemory(m_logicalDevice, m_bufferMemory, nullptr);
}

void VulkanVertexBuffer::bind() const {}
void VulkanVertexBuffer::unbind() const  { /* Vulkan does not have an explicit unbind buffer(s) function */ }

void VulkanVertexBuffer::setData(const void* data, size_t size) {
   void* mappedData;
   vkMapMemory(m_logicalDevice, m_bufferMemory, 0, size, 0, &mappedData);
   memcpy(mappedData, data, size);
   vkUnmapMemory(m_logicalDevice, m_bufferMemory);
}
const BufferLayout& VulkanVertexBuffer::getLayout() const { return m_layout; }
void VulkanVertexBuffer::setLayout(const BufferLayout& layout) { m_layout = layout; }


//********** Index Buffer **********//

VulkanIndexBuffer::VulkanIndexBuffer(uint32_t* indices, size_t count) {
   m_count = count;
   createBuffer(count * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ||VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   void* mappedData;
   vkMapMemory(m_logicalDevice, m_bufferMemory, 0, count * sizeof(uint32_t), 0, &mappedData);
   memcpy(mappedData, indices, count * sizeof(uint32_t));
   vkUnmapMemory(m_logicalDevice, m_bufferMemory);
}
VulkanIndexBuffer::~VulkanIndexBuffer() {
   if (m_buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_logicalDevice, m_buffer, nullptr);
   if (m_bufferMemory != VK_NULL_HANDLE) vkFreeMemory(m_logicalDevice, m_bufferMemory, nullptr);
}

void VulkanIndexBuffer::bind() const {}
void VulkanIndexBuffer::unbind() const { /* Vulkan does not have an explicit unbind buffer(s) function */ }

size_t VulkanIndexBuffer::getCount() const { return m_count; }


//********** Uniform Buffer **********//

VulkanUniformBuffer::VulkanUniformBuffer(size_t size) {
   createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}
VulkanUniformBuffer::VulkanUniformBuffer(void* data, size_t size) {
   createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   setData(data, size);
}
VulkanUniformBuffer::~VulkanUniformBuffer() {
   if (m_buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_logicalDevice, m_buffer, nullptr);
   if (m_bufferMemory != VK_NULL_HANDLE) vkFreeMemory(m_logicalDevice, m_bufferMemory, nullptr);
}

void VulkanUniformBuffer::bind() const {}
void VulkanUniformBuffer::unbind() const { /* Vulkan does not have an explicit unbind for buffers */ }

void VulkanUniformBuffer::setData(const void* data, size_t size) {
   void* mappedData;
   vkMapMemory(m_logicalDevice, m_bufferMemory, 0, size, 0, &mappedData);
}
const BufferLayout& VulkanUniformBuffer::getLayout() const { return m_layout; }
void VulkanUniformBuffer::setLayout(const BufferLayout& layout) { m_layout = layout; }


//********** Shader Storage Buffer **********//

VulkanShaderStorageBuffer::VulkanShaderStorageBuffer(size_t size) {
   createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}
VulkanShaderStorageBuffer::VulkanShaderStorageBuffer(void* data, size_t size) {
   createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   setData(data, size);
}
VulkanShaderStorageBuffer::~VulkanShaderStorageBuffer() {
   if (m_buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_logicalDevice, m_buffer, nullptr);
   if (m_bufferMemory != VK_NULL_HANDLE) vkFreeMemory(m_logicalDevice, m_bufferMemory, nullptr);
}

void VulkanShaderStorageBuffer::bind() const {}
void VulkanShaderStorageBuffer::unbind() const { /* Vulkan does not have an explicit unbind buffer(s) function */ }

void VulkanShaderStorageBuffer::setData(const void* data, size_t size) {
   void* mappedData;
   vkMapMemory(m_logicalDevice, m_bufferMemory, 0, size, 0, &mappedData);
}
const BufferLayout& VulkanShaderStorageBuffer::getLayout() const { return m_layout; }
void VulkanShaderStorageBuffer::setLayout(const BufferLayout& layout) { m_layout = layout; }