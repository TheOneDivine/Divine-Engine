#pragma once

#include "Platforms/Debug/PlatformDetection.hpp"

#if defined(ENGINE_PLATFORM_WINDOWS32) || defined(ENGINE_PLATFORM_WINDOWS64)
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

// TODO: need to make memory allocator (HUGE performance gains)
#define DEFAULT_ALLOCATOR nullptr

// TODO: these need to be defined else where
static const char* TITLE = "Divine Engine"; // app title
static const uint32_t WIDTH =  960;         // app width (in pixels)
static const uint32_t HEIGHT = 540;         // app height (in pixels)

// TODO: needs to define the correct Graphics API
static int GetAPI() { return 0; }

#if defined(DEBUG) || defined(RELEASE)
   #define _DEBUGGING_ENABLED true
#else 
   #define _DEBUGGING_ENABLED false
#endif

#if defined(ENGINE_PLATFORM_WINDOWS32) || defined(ENGINE_PLATFORM_WINDOWS64)
   #ifndef WINDOWS_H
      #define WINDOWS_H
      #include <Windows.h>

   #endif// WINDOWS_H
#endif