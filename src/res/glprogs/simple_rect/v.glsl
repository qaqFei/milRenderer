#version 330 core

in vec2 in_pos;
in vec2 in_uv;
uniform vec4 trans_abcd;
uniform vec2 trans_ef;
uniform vec2 viewport;

vec2 transv2(vec2 p, vec2 scale) {
    p.y *= -1;
    p /= 2;
    p *= scale;
    p += scale / 2;

    vec2 res = vec2(
        p.x * trans_abcd.x + p.y * trans_abcd.z + trans_ef.x,
        p.x * trans_abcd.y + p.y * trans_abcd.w + trans_ef.y
    );

    res -= scale / 2;
    res /= scale;
    res *= 2;
    res.y *= -1;
    return res;
}

void main(){
    vec2 t_in_pos = transv2(in_pos, viewport);
    gl_Position = vec4(t_in_pos, 0.0, 1.0);
}
