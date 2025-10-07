g++ -shared -fPIC -O3 -mavx2 -mfma -fopenmp -g -o libSimpleVideoWriter.so libSimpleVideoWriter.cpp \
    -lavcodec -lavformat -lavutil -lswscale