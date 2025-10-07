#version 330

in vec2 a_pos; // pixel position
in vec2 a_uv;
out vec2 v_uv;

uniform vec2 u_vp;  // viewport

void main(){
    vec2 ndc = (a_pos / u_vp) * 2.0 - 1.0;
    gl_Position = vec4(ndc * vec2(1,-1), 0.0, 1.0);
    v_uv = a_uv;
}
