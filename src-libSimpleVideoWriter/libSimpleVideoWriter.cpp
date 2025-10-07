#include "libSimpleVideoWriter.h"

VideoContext* CreateVideoContext(i64 width, i64 height, f64 frameRate) {
    VideoContext* ctx = new VideoContext();
    ctx->width = width;
    ctx->height = height;
    ctx->frameRate = frameRate;
    ctx->frameIndex = 0;

    ctx->formatCtx = avformat_alloc_context();
    const AVOutputFormat* fmt = av_guess_format("mp4", nullptr, nullptr);
    ctx->formatCtx->oformat = fmt;

    return ctx;
}

static const char* av_err2str_cpp(int err) {
    static char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, buf, sizeof(buf));
    return buf;
}

bool InitializeVideoContext(
    VideoContext* ctx, const char* path,
    const char* vCodecName, const char* aCodecName,
    bool hasAudio,
    f32* audioData, i64 aSampleRate, i64 aChannels, i64 aNumFrames,
    i64 aBitRate
) {
    const AVCodec* vCodec = avcodec_find_encoder_by_name(vCodecName);
    if (!vCodec) return false;

    ctx->stream = avformat_new_stream(ctx->formatCtx, vCodec);
    ctx->codecCtx = avcodec_alloc_context3(vCodec);
    ctx->codecCtx->width = ctx->width;
    ctx->codecCtx->height = ctx->height;
    ctx->codecCtx->time_base = {1, (i32)ctx->frameRate};
    ctx->codecCtx->framerate = {(i32)ctx->frameRate, 1};
    ctx->codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->codecCtx->gop_size = 10;
    ctx->codecCtx->max_b_frames = 1;

    avcodec_parameters_from_context(ctx->stream->codecpar, ctx->codecCtx);
    if (avcodec_open2(ctx->codecCtx, vCodec, nullptr) < 0) return false;

    ctx->frame = av_frame_alloc();
    ctx->frame->format = ctx->codecCtx->pix_fmt;
    ctx->frame->width  = ctx->codecCtx->width;
    ctx->frame->height = ctx->codecCtx->height;
    av_frame_get_buffer(ctx->frame, 0);
    ctx->packet = av_packet_alloc();

    ctx->hasAudio = hasAudio;

    if (ctx->hasAudio) {
        const AVCodec* aCodec = avcodec_find_encoder_by_name(aCodecName);
        if (!aCodec) { fprintf(stderr,"no AAC encoder\n"); return false; }

        ctx->aStream = avformat_new_stream(ctx->formatCtx, aCodec);
        ctx->aCodecCtx = avcodec_alloc_context3(aCodec);

        ctx->aCodecCtx->sample_fmt = AV_SAMPLE_FMT_FLTP;
        ctx->aCodecCtx->bit_rate = aBitRate;
        ctx->aCodecCtx->sample_rate = (i32)aSampleRate;
        ctx->aCodecCtx->ch_layout.nb_channels = (i32)aChannels;
        av_channel_layout_default(&ctx->aCodecCtx->ch_layout, aChannels);
        ctx->aCodecCtx->time_base = {1, ctx->aCodecCtx->sample_rate};

        if (avcodec_open2(ctx->aCodecCtx, aCodec, nullptr) < 0) return false;
        avcodec_parameters_from_context(ctx->aStream->codecpar, ctx->aCodecCtx);
        ctx->aStream->time_base = ctx->aCodecCtx->time_base;

        ctx->audioPts = 0;
    }

    int ret = 0;
    if (!(ctx->formatCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ctx->formatCtx->pb, path, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "[CreateVideoctx] avio_open failed: %s\n", av_err2str_cpp(ret));
            return false;
        }
    }
    ret = avformat_write_header(ctx->formatCtx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[CreateVideoctx] avformat_write_header failed: %s\n", av_err2str_cpp(ret));
        return false;
    }

    if (ctx->hasAudio) {
        const int frameSize = ctx->aCodecCtx->frame_size;

        for (i64 offset = 0; offset + frameSize <= aNumFrames; offset += frameSize)
        {
            AVFrame* f = av_frame_alloc();
            f->format = ctx->aCodecCtx->sample_fmt;
            av_channel_layout_copy(&f->ch_layout, &ctx->aCodecCtx->ch_layout);
            f->sample_rate = ctx->aCodecCtx->sample_rate;
            f->nb_samples = frameSize;
            f->ch_layout.nb_channels = ctx->aCodecCtx->ch_layout.nb_channels;

            int r = av_frame_get_buffer(f, 0);
            if (r < 0) {
                fprintf(stderr, "[PutAudioIntoVideoctx] av_frame_get_buffer failed: %s\n", av_err2str_cpp(r));
                av_frame_free(&f);
                continue;
            }

            for (i64 c = 0; c < f->ch_layout.nb_channels; ++c) {
                f32* data = (f32*)f->data[c];
                for (i64 i = 0; i < frameSize; ++i) {
                    data[i] = audioData[(i + offset) * aChannels + c];
                }
            }

            f->pts = ctx->audioPts;
            ctx->audioPts += frameSize;
            avcodec_send_frame(ctx->aCodecCtx, f);

            if (offset + frameSize > aNumFrames) {
                avcodec_send_frame(ctx->aCodecCtx, nullptr);
            }
            
            while (avcodec_receive_packet(ctx->aCodecCtx, ctx->packet) == 0) {
                av_packet_rescale_ts(ctx->packet, ctx->aCodecCtx->time_base, ctx->aStream->time_base);
                ctx->packet->stream_index = ctx->aStream->index;
                av_interleaved_write_frame(ctx->formatCtx, ctx->packet);
                av_packet_unref(ctx->packet);
            }

            av_frame_free(&f);
        }
    }

    return true;
}

void ReleaseVideoContext(VideoContext* ctx) {
    AVFormatContext* fmt = ctx->formatCtx;

    int ret = avcodec_send_frame(ctx->codecCtx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[ReleaseVideoctx] send_frame(NULL) failed: %s\n", av_err2str_cpp(ret));
    }

    while ((ret = avcodec_receive_packet(ctx->codecCtx, ctx->packet)) == 0) {
        av_packet_rescale_ts(ctx->packet, ctx->codecCtx->time_base, ctx->stream->time_base);
        ret = av_interleaved_write_frame(fmt, ctx->packet);
        if (ret < 0) {
            fprintf(stderr, "[ReleaseVideoctx] write_frame failed: %s\n", av_err2str_cpp(ret));
        }
        av_packet_unref(ctx->packet);
    }

    av_write_trailer(fmt);

    if (!(fmt->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt->pb);

    avcodec_free_context(&ctx->codecCtx);
    av_frame_free(&ctx->frame);
    av_packet_free(&ctx->packet);
    sws_freeContext(ctx->swsCtx);
    avformat_free_context(fmt);

    ctx->formatCtx = nullptr;
    ctx->codecCtx = nullptr;
    ctx->frame = nullptr;
    ctx->packet = nullptr;
    ctx->swsCtx = nullptr;
}

void PutFrame(VideoContext* ctx, iu8* rgbBuffer, i64 width, i64 height) {
    i64 pxCount = width * height;

    if (!ctx->swsCtx) {
        ctx->swsCtx = sws_getContext(
            width, height, AV_PIX_FMT_RGB24,
            width, height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
    }

    AVFrame* rgbFrame = av_frame_alloc();
    av_image_alloc(rgbFrame->data, rgbFrame->linesize, width, height, AV_PIX_FMT_RGB24, 1);

    for (int y = 0; y < height; ++y) {
        memcpy(rgbFrame->data[0] + y * rgbFrame->linesize[0], rgbBuffer + y * width * 3, width * 3);
    }

    sws_scale(ctx->swsCtx, (const iu8* const*)rgbFrame->data, rgbFrame->linesize, 0, height, ctx->frame->data, ctx->frame->linesize);

    ctx->frame->pts = ctx->frameIndex++;

    int ret = avcodec_send_frame(ctx->codecCtx, ctx->frame);
    if (ret < 0) goto END;

    while (ret >= 0) {
        ret = avcodec_receive_packet(ctx->codecCtx, ctx->packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        av_packet_rescale_ts(ctx->packet, ctx->codecCtx->time_base, ctx->stream->time_base);
        av_interleaved_write_frame(ctx->formatCtx, ctx->packet);
        av_packet_unref(ctx->packet);
    }

END:
    av_freep(&rgbFrame->data[0]);
    av_frame_free(&rgbFrame);
}

char* GetEncoders(bool isVideo) {
    const AVCodec* c = nullptr;
    void* opaque = nullptr;
    std::string res;

    while ((c = av_codec_iterate(&opaque))) {
        if (!av_codec_is_encoder(c)) continue;
        if (c->type == AVMEDIA_TYPE_VIDEO && isVideo) {
            res += c->name;
            res += '\0';
        } else if (c->type == AVMEDIA_TYPE_AUDIO && !isVideo) {
            res += c->name;
            res += '\0';
        }
    }

    res += '\0';
    
    char* heap = (char*)malloc(res.size() + 1);
    memcpy(heap, res.data(), res.size());
    heap[res.size()] = '\0';
    return heap;   
}

void FreeString(char* str) {
    free(str);
}

bool HasEncoder(const char* name) {
    if (!name) return false;
    const AVCodec* c = avcodec_find_encoder_by_name(name);
    return c != nullptr;
}
