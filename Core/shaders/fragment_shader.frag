#version 450

layout(binding = 1) uniform sampler2D imageSampler;

layout(location = 0) in vec3 i_fragColor; // output from vertex shader
layout(location = 1) in vec2 i_texCoord;  // output from vertex shader

layout(location = 0) out vec4 o_color;

void main() {
     o_color = texture(imageSampler, i_texCoord);
}