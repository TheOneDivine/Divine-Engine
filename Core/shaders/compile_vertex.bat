E:\libraries\Vulkan_SDK\Bin\glslc.exe shaders\vertex_shader.vert -o shaders\vert.spv

E:\libraries\Vulkan_SDK\Bin\spirv-opt.exe --merge-return --inline-entry-points-exhaustive --eliminate-dead-functions --scalar-replacement --eliminate-local-single-block --eliminate-local-single-store --simplify-instructions --vector-dce --eliminate-dead-inserts --eliminate-dead-code-aggressive --eliminate-dead-branches --merge-blocks --eliminate-local-multi-store --simplify-instructions --vector-dce --eliminate-dead-inserts --redundancy-elimination  --eliminate-dead-code-aggressive --strip-debug -o shaders\optimized_vert.spv shaders\vert.spv