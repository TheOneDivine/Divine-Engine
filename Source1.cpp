
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
if (vkCreateBuffer(logicalDevice, &bufferInfo, DEFAULT_ALLOCATOR, &buffer) != VK_SUCCESS)
throw std::runtime_error("failed to create buffer!");

// get buffer memory requirements
VkMemoryRequirements memRequirements;
vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

// allocate memory for buffer
VkMemoryAllocateInfo allocInfo = {
   .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
   .pNext = nullptr,
   .allocationSize = memRequirements.size,
   .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
};
if (vkAllocateMemory(logicalDevice, &allocInfo, DEFAULT_ALLOCATOR, &bufferMemory) != VK_SUCCESS)
throw std::runtime_error("failed to allocate buffer memory!");

// bind memory to buffer
vkBindBufferMemory(m_logicalDevice, buffer, bufferMemory, 0);



// create a staging buffer for the vertex data
VkBuffer stagingBuffer;
VkDeviceMemory stagingBufferMemory;

// create buffer//

VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = bufferSize,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr
};
if (vkCreateBuffer(logicalDevice, &bufferInfo, DEFAULT_ALLOCATOR, &buffer) != VK_SUCCESS)
throw std::runtime_error("failed to create buffer!");

// get buffer memory requirements
VkMemoryRequirements memRequirements;
vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

// allocate memory for buffer
VkMemoryAllocateInfo allocInfo = {
   .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
   .pNext = nullptr,
   .allocationSize = memRequirements.bufferSize,
   .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
};
if (vkAllocateMemory(logicalDevice, &allocInfo, DEFAULT_ALLOCATOR, &bufferMemory) != VK_SUCCESS)
throw std::runtime_error("failed to allocate buffer memory!");

// bind memory to buffer
vkBindBufferMemory(m_logicalDevice, buffer, bufferMemory, 0);

// create buffer end

createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

// copy the vertex data to the staging buffer
void* data;
vkMapMemory(ogicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
memcpy(data, data, (size_t)bufferSize);
vkUnmapMemory(logicalDevice, stagingBufferMemory);

// create buffer and map vertices to it
createDescriptorBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TYPE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);
copyDescriptorBuffer(stagingBuffer, buffer, bufferSize, queue);

// clean up the staging buffer
vkDestroyBuffer(logicalDevice, stagingBuffer, DEFAULT_ALLOCATOR);
vkFreeMemory(logicalDevice, stagingBufferMemory, DEFAULT_ALLOCATOR);