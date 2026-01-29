#pragma once

#ifdef _WIN32// 32-bit Windows
   #ifdef _WIN64// 64-bit Windows
      #define ENGINE_PLATFORM_WINDOWS64
   #else
      #define ENGINE_PLATFORM_WINDOWS32
      #error "x86 Builds are not supported yet!"

   #endif// 64-bit Windows

#elif defined(__APPLE__) || defined(__MACH__)
   #include <TargetConditionals.h>
   /* TARGET_OS_MAC exists on all the platforms
    * so we must check all of them (in this order)
    * to ensure that we're running on MAC
    * and not some other Apple platform */
   #if TARGET_IPHONE_SIMULATOR == 1
      #error "IOS simulator is not supported yet!"

   #elif TARGET_OS_IPHONE == 1
      #define ENGINE_PLATFORM_IOS
      #error "IOS is not supported yet!"

   #elif TARGET_OS_MAC == 1
      #define ENGINE_PLATFORM_MACOS
      #error "MacOS is not supported yet!"

   #else
      #error "Unknown Apple platform yet!"

   #endif
    /* We also have to check __ANDROID__ before __linux__
     * since android is based on the linux kernel
     * it has __linux__ defined */
#elif defined(__ANDROID__)
   #define ENGINE_PLATFORM_ANDROID
   #error "Android is not supported yet!"

#elif defined(__linux__)
   #define ENGINE_PLATFORM_LINUX
   #error "Linux is not supported yet!"

#else

#error "Unknown platform!"
#endif // End of platform detection