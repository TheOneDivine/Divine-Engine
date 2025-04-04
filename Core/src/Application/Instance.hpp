#pragma once

#include "../preComp.hpp"

// Graphics API instance
class GAPIInstance
{
public:
   static void* Get();
   static void Destroy();
};