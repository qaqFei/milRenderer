#include "libSimpleVideoWriter.h"

VideoContext* CreateVideoCap(i64 width, i64 height, f64 frameRate) {
    VideoCap* cap = new VideoCap();
    cap->width = width;
    cap->height = height;
    cap->frameRate = frameRate;
    cap->frameIndex = 0;

    cap->formatCtx = avformat_alloc_context();
    const AVOutputFormat* fmt = av_guess_format("mp4", nullptr, nullptr);
    cap->formatCtx->oformat = fmt;

    return cap;
}
