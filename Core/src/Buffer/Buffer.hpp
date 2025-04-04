#pragma once

#include "../preComp.hpp"

#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

#include <string>
#include <sstream>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>

enum class ShaderDataType
{
   None = 0,

   // scalars

   Int, Uint, Float, Double, Bool,

   // vectors

   Ivec2, Ivec3, Ivec4, // int
   Uvec2, Uvec3, Uvec4, // unsigned int
   Vec2, Vec3, Vec4,    // float
   Dvec2, Dvec3, Dvec4, // double
   Bvec2, Bvec3, Bvec4, // bool

   // float matrices

   Mat2, Mat3, Mat4,       // short hands
   Mat2x2, Mat2x3, Mat2x4, // 2x
   Mat3x2, Mat3x3, Mat3x4, // 3x
   Mat4x2, Mat4x3, Mat4x4, // 4x

   // double matrices

   Dmat2, Dmat3, Dmat4,       // short hands
   Dmat2x2, Dmat2x3, Dmat2x4, // 2x
   Dmat3x2, Dmat3x3, Dmat3x4, // 3x
   Dmat4x2, Dmat4x3, Dmat4x4, // 4x

};

static uint32_t ShaderDataTypeSize(ShaderDataType type)
{
   switch (type)
   {
      // scalars

      case ShaderDataType::Int:     return 4; // int
      case ShaderDataType::Uint:    return 4; // unsigned int
      case ShaderDataType::Float:   return 4; // float
      case ShaderDataType::Double:  return 8; // double
      case ShaderDataType::Bool:    return 4; // bool

      // vectors

      case ShaderDataType::Ivec2:   return 8;  // 2 * int
      case ShaderDataType::Ivec3:   return 16; // 3 * int + 4 bytes padding
      case ShaderDataType::Ivec4:   return 16; // 4 * int
      case ShaderDataType::Uvec2:   return 8;  // 2 * unsigned int
      case ShaderDataType::Uvec3:   return 16; // 3 * unsigned int + 4 bytes padding
      case ShaderDataType::Uvec4:   return 16; // 4 * unsigned int
      case ShaderDataType::Vec2:    return 8;  // 2 * float
      case ShaderDataType::Vec3:    return 16; // 3 * float + 4 bytes padding
      case ShaderDataType::Vec4:    return 16; // 4 * float
      case ShaderDataType::Dvec2:   return 16; // 2 * double
      case ShaderDataType::Dvec3:   return 32; // 3 * double + 8 bytes padding
      case ShaderDataType::Dvec4:   return 32; // 4 * double
      case ShaderDataType::Bvec2:   return 8;  // 2 * bool
      case ShaderDataType::Bvec3:   return 16; // 3 * bool + 4 bytes padding
      case ShaderDataType::Bvec4:   return 16; // 4 * bool

      // float matrices

      case ShaderDataType::Mat2:    return 16; // 2 * Vec2
      case ShaderDataType::Mat3:    return 48; // 3 * Vec3
      case ShaderDataType::Mat4:    return 64; // 4 * Vec4
      case ShaderDataType::Mat2x2:  return 16; // 2 * Vec2
      case ShaderDataType::Mat2x3:  return 32; // 2 * Vec3
      case ShaderDataType::Mat2x4:  return 32; // 2 * Vec4
      case ShaderDataType::Mat3x2:  return 24; // 3 * Vec2
      case ShaderDataType::Mat3x3:  return 48; // 3 * Vec3
      case ShaderDataType::Mat3x4:  return 48; // 3 * Vec4
      case ShaderDataType::Mat4x2:  return 32; // 4 * Vec2
      case ShaderDataType::Mat4x3:  return 64; // 4 * Vec3
      case ShaderDataType::Mat4x4:  return 64; // 4 * Vec4

      // double matrices

      case ShaderDataType::Dmat2:   return 32;  // 2 * Dvec2
      case ShaderDataType::Dmat3:   return 64;  // 3 * Dvec3
      case ShaderDataType::Dmat4:   return 128; // 4 * Dvec4
      case ShaderDataType::Dmat2x2: return 32;  // 2 * Dvec2
      case ShaderDataType::Dmat2x3: return 64;  // 2 * Dvec3
      case ShaderDataType::Dmat2x4: return 64;  // 2 * Dvec4
      case ShaderDataType::Dmat3x2: return 48;  // 3 * Dvec2
      case ShaderDataType::Dmat3x3: return 96;  // 3 * Dvec3
      case ShaderDataType::Dmat3x4: return 96;  // 3 * Dvec4
      case ShaderDataType::Dmat4x2: return 64;  // 4 * Dvec2
      case ShaderDataType::Dmat4x3: return 128; // 4 * Dvec3
      case ShaderDataType::Dmat4x4: return 128; // 4 * Dvec4
   }

   return 0;
}

struct BufferElement
{
   std::string name;
   ShaderDataType type;
   uint32_t size;
   size_t offset;
   bool norm;

   BufferElement() = default;

   BufferElement(ShaderDataType type, const std::string& name, bool normalized = false) : name(name), type(type), size(ShaderDataTypeSize(type)), offset(0), norm(normalized) {}

   uint32_t GetComponentCount() const {
      switch (type)
      {
         // scalars

         case ShaderDataType::Int:     return 1;
         case ShaderDataType::Uint:    return 1;
         case ShaderDataType::Float:   return 1;
         case ShaderDataType::Double:  return 1;
         case ShaderDataType::Bool:    return 1;
         
         // vectors
         
         case ShaderDataType::Ivec2:   return 2;
         case ShaderDataType::Ivec3:   return 3;
         case ShaderDataType::Ivec4:   return 4;
         case ShaderDataType::Uvec2:   return 2;
         case ShaderDataType::Uvec3:   return 3;
         case ShaderDataType::Uvec4:   return 4;
         case ShaderDataType::Vec2:    return 2;
         case ShaderDataType::Vec3:    return 3;
         case ShaderDataType::Vec4:    return 4;
         case ShaderDataType::Dvec2:   return 2;
         case ShaderDataType::Dvec3:   return 3;
         case ShaderDataType::Dvec4:   return 4;
         case ShaderDataType::Bvec2:   return 2;
         case ShaderDataType::Bvec3:   return 3;
         case ShaderDataType::Bvec4:   return 4;
         
         // float matrices
         
         case ShaderDataType::Mat2:    return 4;
         case ShaderDataType::Mat3:    return 9;
         case ShaderDataType::Mat4:    return 16;
         case ShaderDataType::Mat2x2:  return 4;
         case ShaderDataType::Mat2x3:  return 6;
         case ShaderDataType::Mat2x4:  return 8;
         case ShaderDataType::Mat3x2:  return 6;
         case ShaderDataType::Mat3x3:  return 9;
         case ShaderDataType::Mat3x4:  return 12;
         case ShaderDataType::Mat4x2:  return 8;
         case ShaderDataType::Mat4x3:  return 12;
         case ShaderDataType::Mat4x4:  return 16;
         
         // double matrices
         
         case ShaderDataType::Dmat2:   return 4;
         case ShaderDataType::Dmat3:   return 9;
         case ShaderDataType::Dmat4:   return 16;
         case ShaderDataType::Dmat2x2: return 4;
         case ShaderDataType::Dmat2x3: return 6;
         case ShaderDataType::Dmat2x4: return 8;
         case ShaderDataType::Dmat3x2: return 6;
         case ShaderDataType::Dmat3x3: return 9;
         case ShaderDataType::Dmat3x4: return 12;
         case ShaderDataType::Dmat4x2: return 8;
         case ShaderDataType::Dmat4x3: return 12;
         case ShaderDataType::Dmat4x4: return 16;
      }

      return 0;
   }
};

class BufferLayout
{
public:
   BufferLayout() {}

   BufferLayout(std::initializer_list<BufferElement> elements) : m_elements(elements) {
      CalculateOffsetsAndStride();
   }

   uint32_t GetStride() const { return m_stride; }
   const std::vector<BufferElement>& GetElements() const { return m_elements; }

   std::vector<BufferElement>::iterator begin() { return m_elements.begin(); }
   std::vector<BufferElement>::iterator end() { return m_elements.end(); }
   std::vector<BufferElement>::const_iterator begin() const { return m_elements.begin(); }
   std::vector<BufferElement>::const_iterator end() const { return m_elements.end(); }
private:

   void CalculateOffsetsAndStride() {
      size_t offset = 0;
      m_stride = 0;
      for (auto& element : m_elements) {
         element.offset = offset;
         offset += element.size;
         m_stride += element.size;
      }
   }
private:
   std::vector<BufferElement> m_elements;
   uint32_t m_stride = 0;
};

class VertexBuffer
{
public:
   virtual ~VertexBuffer() = default;

   virtual void bind() const = 0;
   virtual void unbind() const = 0;

   virtual void setData(const void* data, size_t size) = 0;

   virtual const BufferLayout& getLayout() const = 0;
   virtual void setLayout(const BufferLayout& layout) = 0;

   static Ref<VertexBuffer> create(size_t size);
   static Ref<VertexBuffer> create(void* vertices, size_t size);
};

class IndexBuffer
{
public:
   virtual ~IndexBuffer() = default;

   virtual size_t getCount() const = 0;

   virtual void bind() const = 0;
   virtual void unbind() const = 0;

   static Ref<IndexBuffer> create(uint32_t* indeces, size_t size);
};

class UniformBuffer
{
public:
   virtual ~UniformBuffer() = default;

   virtual void bind() const = 0;
   virtual void unbind() const = 0;

   virtual void setData(const void* data, size_t size) = 0;

   virtual const BufferLayout& getLayout() const = 0;
   virtual void setLayout(const BufferLayout& layout) = 0;

   static Ref<UniformBuffer> create(size_t size);
   static Ref<UniformBuffer> create(void* data, size_t size);
};

class ShaderStorageBuffer
{
public:
   virtual ~ShaderStorageBuffer() = default;

   virtual void bind() const = 0;
   virtual void unbind() const = 0;

   virtual void setData(const void* data, size_t size) = 0;

   virtual const BufferLayout& getLayout() const = 0;
   virtual void setLayout(const BufferLayout& layout) = 0;

   static Ref<ShaderStorageBuffer> create(size_t size);
   static Ref<ShaderStorageBuffer> create(void* data, size_t size);
};