import moderngl as mgl
from PIL import Image
import numpy as np

ctx = mgl.create_context(standalone=True, backend="egl")
w, h = 800, 600
ctx.viewport = (0, 0, w, h)

prog = ctx.program(vertex_shader=open("./res/glprogs/simple_texture/v.glsl").read(),
                   fragment_shader=open("./res/glprogs/simple_texture/f.glsl").read())

fbo = ctx.simple_framebuffer((w, h), 4, dtype="u1")
fbo.use()
texim = Image.open("./res/extap.png")
tex = ctx.texture(texim.size, 4, texim.tobytes(), dtype="u1")

x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
quad = np.array([
    x1,  y2,  0.0, 1.0,
    x2,  y2,  1.0, 1.0,
    x1,  y1,  0.0, 0.0,
    x2,  y1,  1.0, 0.0,
], dtype='f4')
vertices = ctx.buffer(quad.tobytes())
ibo = ctx.buffer(np.array([0, 1, 2, 1, 3, 2], dtype='u1').tobytes())
vao = ctx.vertex_array(
    prog,
    [(vertices, '2f 2f', 'in_pos', 'in_uv')],
    index_buffer=ibo,
    index_element_size=1
)
# ✅ 绑定纹理到采样器
tex.use(location=0)
prog['u_tex'].value = 0
vao.render(mgl.TRIANGLES)

img = Image.frombytes('RGBA', (w, h), fbo.read(viewport=(0, 0, w, h), components=4, dtype='u1'))
img.save('./screenshot.png')