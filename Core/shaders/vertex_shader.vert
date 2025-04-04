#version 450

layout(location = 0) in vec3 i_pos;       // input vertex position
layout(location = 1) in vec3 i_Color;     // input 
layout(location = 2) in vec2 i_texCoord;  // input vertex normal coordinate

layout(location = 0) out vec3 o_fragColor; // output pixelcolor for fragment shader
layout(location = 1) out vec2 o_texCoord;  // output vertex normal coordinate to fragment shader

void main() {
   gl_Position = vec4(i_pos, 1.0);

   o_fragColor = i_Color;
   o_texCoord = i_texCoord;
}
