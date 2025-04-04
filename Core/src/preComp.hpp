#pragma once

#include "Platforms/Debug/PlatformDetection.hpp"

#if defined(_USING_WINDOWS32) || defined(_USING_WINDOWS64)
   #ifndef NOMINMAX
      // See github.com/skypjack/entt/wiki/Frequently-Asked-Questions#warning-c4003-the-min-the-max-and-the-macro
      #define NOMINMAX
   #endif
#endif

// Standard Library //
#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

// Data Structures //
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "Platforms/Debug/Base.hpp"

// need to make memory allocator (HUGE performance gains)
#define DEFAULT_ALLOCATOR nullptr

// these need to be defined else where
static const char* TITLE = "Hello Triangle"; // app title
static const uint32_t WIDTH = 800;          // app width (in pixels)
static const uint32_t HEIGHT = 400;         // app height (in pixels)

#if defined(_USING_WINDOWS32) || defined(_USING_WINDOWS64)
   #ifndef WINDOWS_H
      #define WINDOWS_H
      #include <Windows.h>

      #endif// WINDOWS_H
#endif