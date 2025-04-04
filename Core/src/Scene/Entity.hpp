#pragma once

#include "../preComp.hpp"

#include "Components.hpp"
#include "Scene.hpp"
#include "UUID.hpp"

#include "entt.hpp"

class Entity
{
public:
   Entity() = default;
   Entity(entt::entity handle, Scene* scene);

   Entity(const Entity& other) = default;
   ~Entity() = default;

   template<typename ComponentType, typename... ComponentArgs>
   ComponentType& addComponent(ComponentArgs&&... args) {
      return m_scene->m_registry.emplace<ComponentType>(m_entity, std::forward<ComponentArgs>(args)...);
   }

   template<typename ComponentType>
   ComponentType& getComponent() {
      return m_scene->m_registry.get<ComponentType>(m_entity);
   }

   template<typename ComponentType>
   bool hasComponent() {
      return m_scene->m_registry.has<ComponentType>(m_entity);
   }

   template<typename ComponentType>
   void removeComponent() {
      m_scene->m_registry.remove<ComponentType>(m_entity);
   }

   operator bool() const { return m_entity != entt::null; }
   operator entt::entity() const { return m_entity; }
   operator uint32_t() const { return (uint32_t)m_entity; }

   bool operator==(const Entity& other) const { return m_entity == other.m_entity && m_scene == other.m_scene; }
   bool operator!=(const Entity& other) const { return !(*this == other); }

private:
   entt::entity m_entity{ entt::null };
   Scene* m_scene = nullptr;
};