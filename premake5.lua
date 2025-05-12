workspace "Divine_Engine"

   configurations {
      -- plain source code, full debug outputs
      "Debug",
      -- optimized source code, most debug outputs
      "Release",
      -- fully optimized, no debugging
      "Distribute"
   }
   
   architecture "x86_64"

   filter {}

project "Core"
   location "Core"
   kind "ConsoleApp"
   language "C++"

   outputdir = "%{prj.name}/%{cfg.system}/%{cfg.architecture}/%{cfg.buildcfg}"

   --E.g. Binary/Core/windows/x64/debug/Core.exe
   targetdir ("Binary/" .. outputdir)
   objdir ("Binary_Obj/" .. outputdir)

   pchheader "preComp.hpp"
	pchsource "src/preComp.cpp"

   files {
      "%{prj.name}/src/shaders/**.glsl",
      "%{prj.name}/src/shaders/**.hlsl",
      "%{prj.name}/src/**.hpp",
      "%{prj.name}/src/**.cpp",
      "%{prj.name}/src/**.h",
      "%{prj.name}/src/**.c",
   }

   --absolute paths need to be fixed
   includedirs {
      "C:/Users/coold/Dev/Library/Vulkan/Include",
      "C:/Users/coold/Dev/Library/glfw-3.4.WIN64/include",
      "C:/Users/coold/Dev/Library/glm-master",
      "C:/Users/coold/Dev/Library/stb-master",
      "C:/Users/coold/Dev/Library/tinyobjloader-release"
   }

   libdirs {
      "C:/Users/coold/Dev/Library/Vulkan/Lib",
      "C:/Users/coold/Dev/Library/glfw-3.4.WIN64/lib-vc2022"
   }

   links {
      "vulkan-1",
      "glfw3"
   }

   filter { "system:windows" }
      systemversion "latest"
      cppdialect "C++20"
      cdialect "c17"

   filter { "configurations:Debug" }
      optimize "Off"
      symbols "On"
      defines { "DEBUG" }

   filter { "configurations:Release" }
      optimize "On"
      symbols "On"
      defines { "RELEASE" }

   filter { "configurations:Distribute" }
      optimize "Speed" -- if exacutible has exsesive bloating, switch to Full
      symbols "Off"
      defines { "DISTRIBUTE" }

   filter {}