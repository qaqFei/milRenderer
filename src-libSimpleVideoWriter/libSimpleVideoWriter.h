#define i64 long
#define i32 int
#define f64 double

#include <cstdio>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}

struct VideoContext {
    AVFormatContext* formatCtx;
    AVCodecContext* codecCtx;
    AVStream* stream;
    AVFrame* frame;
    AVPacket* packet;
    SwsContext* swsCtx;
    i32 frameIndex;

    bool hasAudio;
    AVStream* aStream;
    AVCodecContext* aCodecCtx;
    i64 audioPts;
}

extern "C" {
    VideoContext* CreateVideoCap(i64 width, i64 height, f64 frameRate)
}
