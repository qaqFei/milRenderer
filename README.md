# milRenderer

使用 Python&C++ 编写的 Milthm 自制谱渲染器。

## 从源码运行

测试系统: `Ubuntu 24.04.2 LTS`

```bash
git clone https://github.com/qaqFei/milRenderer.git --depth 1

cd milRenderer

cd src-libSimpleVideoWriter
bash install_deps.sh
bash compile.sh
bash dev_install.sh

cd ../src
pip install -r requirements.txt

python main.py -i /path/to/milthm/chart/file -o /path/to/output/file
```

### 踩坑

如果你的系统环境正常, 但是出现 `Exception: requested device index 0, but found 0 devices`,

特别是在安装 `ffmpeg` 之后,

可以尝试 `sudo apt install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev`
