git clone https://github.com/qaqFei/ffmpeg-8.0-source --depth 1
cd ffmpeg-8.0-source
unzip ffmpeg-8.0.zip -d ./ffmpeg-8.0
unzip nv-codec-headers.zip -d ./nv-codec-headers

cd ./nv-codec-headers
sudo make install

# chmod +x ./configure
# sudo apt install nasm -y
# ./configure --enable-cuda-nvcc --enable-cuda-sdk --enable-libnpp --enable-nvenc --extra-cflags="-I/usr/local/cuda/include" --extra-ldflags="-L/usr/local/cuda/lib64" --enable-nonfree

./configure --prefix=/usr/local/ffmpeg --enable-shared \
--enable-nonfree --enable-gpl --enable-version3 \
--enable-libmp3lame --enable-libvpx --enable-libopus \
--enable-opencl --enable-libxcb --enable-avresample\
--enable-opengl --enable-nvenc --enable-vaapi \
--enable-vdpau --enable-ffplay --enable-ffprobe \
--enable-libxvid \
--enable-libx264 --enable-libx265 --enable-openal \
--enable-openssl --enable-cuda-nvcc --enable-cuvid --extra-cflags=-I/usr/local/cuda-12.2/include --extra-ldflags=-L/usr/local/cuda-12.2/lib64