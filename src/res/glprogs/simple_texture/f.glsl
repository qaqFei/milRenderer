#version 330 core

in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 f_color;

void main(){
    f_color = texture(u_tex, v_uv);
}
