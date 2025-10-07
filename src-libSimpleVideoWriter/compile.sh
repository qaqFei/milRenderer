g++ -shared -fPIC -O3 -g -o libSimpleVideoWriter.so libSimpleVideoWriter.cpp \
    -lavcodec -lavformat -lavutil -lswscale