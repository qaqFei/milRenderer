#define i64 long
#define i32 int
#define i16 short
#define iu8 unsigned char
#define f64 double
#define f32 float

#include <cstdio>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}

struct VideoContext {
    i64 width;
    i64 height;
    f64 frameRate;

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
};

extern "C" {
    VideoContext* CreateVideoContext(i64 width, i64 height, f64 frameRate);
    bool InitializeVideoContext(VideoContext* ctx, const char* path, bool hasAudio, f32* audioData, i64 aSampleRate, i64 aChannels, i64 aNumFrames, i64 aBitRate);
    void ReleaseVideoContext(VideoContext* ctx);
    void PutFrame(VideoContext* ctx, iu8* rgbBuffer, i64 width, i64 height);
}
