sudo apt update
sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libyuv-dev -y
sudo apt install libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev -y

git clone https://github.com/cginternals/glbinding.git --depth 1
cd glbinding
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPTION_BUILD_EXAMPLES=OFF -DOPTION_BUILD_TOOLS=OFF
cmake --build build -j$(nproc)
sudo cmake --install build
cd ..
rm -rf ./glbinding
