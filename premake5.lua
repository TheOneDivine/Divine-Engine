-- Checks for specified directory
function requireDir(path, name)
    if not os.isdir(path) then
        error("ERROR: Missing required dependency: " .. name .. " at " .. path)
    end
end

-- Load environment variables
   local VULKAN = os.getenv("VULKAN_SDK")
   local GLFW   = os.getenv("GLFW_DIR")
   local GLM    = os.getenv("GLM_DIR")
   local STB    = os.getenv("STB_DIR")
   local TINY   = os.getenv("TINYOBJ_DIR")

workspace "Divine_Engine"

   configurations {
      -- plain source code, full debug outputs
      "Debug",
      -- optimized source code, some debug outputs
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

   -- Validate directories
   requireDir(VULKAN, "Vulkan SDK")
   requireDir(GLFW, "GLFW")
   requireDir(GLM, "GLM")
   requireDir(STB, "stb")
   requireDir(TINY, "tinyobjloader")

   --absolute paths need to be fixed
   includedirs {
   VULKAN .. "/Include",
   GLFW .. "/include",
   GLM,
   STB,
   TINY
   }

   libdirs {
      VULKAN .. "/Lib",
      GLFW   .. "/lib-vc2022"
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
      optimize "Speed" -- if exacutible has excesive bloating, switch to Full
      symbols "Off"
      defines { "DISTRIBUTE" }