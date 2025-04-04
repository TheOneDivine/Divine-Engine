#pragma once

#include "../preComp.hpp"

#include <unordered_map>
#include <vector>
#include <typeindex>
#include <typeinfo>
#include <memory>

#include "UUID.hpp"

#include "entt.hpp"

class Entity;

class Scene
{
public:
   Scene() = default;
   ~Scene() = default;

   Entity createEntity();

   template<typename... ComponentType>
   auto data() {
      return m_registry.view<ComponentType...>();
   }

   template<typename ComponentType>
   size_t size() {
      auto view = m_registry.view<ComponentType>();
      return view.size();
   }

private:
   entt::registry m_registry;

   friend class Entity;
};