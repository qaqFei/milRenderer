#version 330

in  vec2 v_uv;
out vec4 f_col;
uniform sampler2D u_tex;

void main(){
    f_col = texture(u_tex, v_uv);
    f_col.rgb = vec3(1.0, 0.0, 0.0);
}
