#pragma once

#include "UUID.hpp"
#include <glm/glm.hpp>

struct alignas(8) IDComponent
{
   alignas(8) MY::UUID ID;

   IDComponent() = default;
   IDComponent(const IDComponent&) = default;
};

struct alignas(16) SphereComponent
{
   alignas(4) float radius;
   alignas(16) glm::vec3 center;
   

   SphereComponent(glm::vec3 center = glm::vec3{ 0.0f, 0.0f, 0.0f }, float radius = 10.0f) : center(center), radius(radius) {}
   SphereComponent(const SphereComponent&) = default;
};

struct alignas(16) MaterialComponent
{
   alignas(4) float smoothness;                       // material smoothness, 0 = mirror, 1 = matte, used in METAL materials (defualt = mirror)
   alignas(4) float refractionIndex = 2.42f;                 // material refraction index, 0 = empty space, 1.000003 = air, 2.42 = diamond (default = diamond), higher refractions = more bending of light
   alignas(16) glm::vec3 color; // material color (default = red)

   MaterialComponent(glm::vec3 color = glm::vec3{ 1.0f, 0.0f, 0.0f }, float smoothness = 1.0f, float refractionIndex = -1.0f) : color(color), smoothness(smoothness), refractionIndex(refractionIndex) {}
   MaterialComponent(const MaterialComponent&) = default;
};