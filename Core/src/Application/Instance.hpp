#pragma once

#include "../pch.hpp"

// Graphics API instance
class GAPIInstance
{
public:
   static void* Get();
   static void Destroy();
};