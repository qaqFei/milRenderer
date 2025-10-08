#define i64 long
#define i32 int
#define i16 short
#define iu8 unsigned char
#define f64 double
#define f32 float

#include <cstdio>
#include <string>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}

#include <libyuv.h>

struct VideoContext {
    i64 width;
    i64 height;
    f64 frameRate;

    AVFormatContext* formatCtx;
    AVCodecContext* codecCtx;
    AVStream* stream;
    AVFrame* frame;
    AVPacket* packet;
    i32 frameIndex;

    bool hasAudio;
    AVStream* aStream;
    AVCodecContext* aCodecCtx;
    i64 audioPts;
};

extern "C" {
    VideoContext* CreateVideoContext(i64 width, i64 height, f64 frameRate);
    bool InitializeVideoContext(VideoContext* ctx, const char* path, const char* vCodecName, const char* aCodecName, bool hasAudio, f32* audioData, i64 aSampleRate, i64 aChannels, i64 aNumFrames, i64 aBitRate);
    void ReleaseVideoContext(VideoContext* ctx);
    void PutFrame(VideoContext* ctx, iu8* rgbBuffer, i64 width, i64 height);
    char* GetEncoders(bool isVideo);
    void FreeString(char* str);
    bool HasEncoder(const char* name);
}
