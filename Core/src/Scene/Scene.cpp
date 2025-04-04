#include "Scene.hpp"
#include "Entity.hpp"


Entity Scene::createEntity() {
   Entity entity{ m_registry.create(), this };
   entity.addComponent<IDComponent>();
   return entity;
}