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


#define _STR(x) #x
#define STR(x)  _STR(x)
// #pragma message("__AVX2__ = " STR(__AVX2__))
// #pragma message("__FMA__  = " STR(__FMA__))
// #pragma message("_OPENMP  = " STR(_OPENMP))

#if defined(__AVX2__) && defined(__FMA__) && defined(_OPENMP) && 0 // has bugs
#   define HAS_AVX2_RGB24_TO_YUV420 1
#   include <immintrin.h>
#   include <omp.h>
#   pragma message("HAS_AVX2_RGB24_TO_YUV420 = 1")

static inline int align16(int x) { return x & ~15; }
void rgb24_to_yuv420_avx2(const uint8_t* rgb, int w, int h, uint8_t* y, uint8_t* u, uint8_t* v) {
    const int strideY = w;
    const int strideUV = w >> 1;

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int j = 0; j < h; j += 2) {
        const uint8_t *srcTop = rgb + j * w * 3;
        const uint8_t *srcBot = srcTop + w * 3;
        uint8_t *dstY0 = y + j * strideY;
        uint8_t *dstY1 = dstY0 + strideY;
        uint8_t *dstU   = u + (j >> 1) * strideUV;
        uint8_t *dstV   = v + (j >> 1) * strideUV;

        int i = 0;
        for (; i < align16(w); i += 16) {
            __m256i r0, g0, b0, r1, g1, b1;
            // 0..31
            __m256i rg0 = _mm256_loadu_si256((const __m256i*)(srcTop + 0));
            __m256i gb0 = _mm256_loadu_si256((const __m256i*)(srcTop + 32));
            __m256i br0 = _mm256_loadu_si256((const __m256i*)(srcTop + 64));
            // 32..63
            __m256i rg1 = _mm256_loadu_si256((const __m256i*)(srcBot + 0));
            __m256i gb1 = _mm256_loadu_si256((const __m256i*)(srcBot + 32));
            __m256i br1 = _mm256_loadu_si256((const __m256i*)(srcBot + 64));

            // 把 RGB 拆成 3 个 256 位向量
            b0 = _mm256_or_si256(_mm256_and_si256(rg0, _mm256_set1_epi16(0x00FF)),
                                 _mm256_slli_epi16(_mm256_and_si256(gb0, _mm256_set1_epi16(0xFF00)), -8));
            g0 = _mm256_or_si256(_mm256_srli_epi16(_mm256_and_si256(rg0, _mm256_set1_epi16(0xFF00)), 8),
                                 _mm256_and_si256(gb0, _mm256_set1_epi16(0x00FF)));
            r0 = _mm256_or_si256(_mm256_srli_epi16(_mm256_and_si256(gb0, _mm256_set1_epi16(0xFF00)), 8),
                                 _mm256_and_si256(br0, _mm256_set1_epi16(0x00FF)));

            b1 = _mm256_or_si256(_mm256_and_si256(rg1, _mm256_set1_epi16(0x00FF)),
                                 _mm256_slli_epi16(_mm256_and_si256(gb1, _mm256_set1_epi16(0xFF00)), -8));
            g1 = _mm256_or_si256(_mm256_srli_epi16(_mm256_and_si256(rg1, _mm256_set1_epi16(0xFF00)), 8),
                                 _mm256_and_si256(gb1, _mm256_set1_epi16(0x00FF)));
            r1 = _mm256_or_si256(_mm256_srli_epi16(_mm256_and_si256(gb1, _mm256_set1_epi16(0xFF00)), 8),
                                 _mm256_and_si256(br1, _mm256_set1_epi16(0x00FF)));

            // 16bit → 32bit 零扩展
            __m256i zero = _mm256_setzero_si256();
            __m256i r_lo = _mm256_unpacklo_epi16(r0, zero);
            __m256i r_hi = _mm256_unpackhi_epi16(r0, zero);
            __m256i g_lo = _mm256_unpacklo_epi16(g0, zero);
            __m256i g_hi = _mm256_unpackhi_epi16(g0, zero);
            __m256i b_lo = _mm256_unpacklo_epi16(b0, zero);
            __m256i b_hi = _mm256_unpackhi_epi16(b0, zero);

            // Y = ( 66*R + 129*G +  25*B + 128) >> 8 + 16
            __m256i y0_lo = _mm256_add_epi32(_mm256_add_epi32(
                                _mm256_mullo_epi32(r_lo, _mm256_set1_epi32(66)),
                                _mm256_mullo_epi32(g_lo, _mm256_set1_epi32(129))),
                                _mm256_mullo_epi32(b_lo, _mm256_set1_epi32(25)));
            y0_lo = _mm256_add_epi32(y0_lo, _mm256_set1_epi32(128));
            y0_lo = _mm256_srli_epi32(y0_lo, 8);
            y0_lo = _mm256_add_epi32(y0_lo, _mm256_set1_epi32(16));
            __m256i y0_hi = _mm256_add_epi32(_mm256_add_epi32(
                                _mm256_mullo_epi32(r_hi, _mm256_set1_epi32(66)),
                                _mm256_mullo_epi32(g_hi, _mm256_set1_epi32(129))),
                                _mm256_mullo_epi32(b_hi, _mm256_set1_epi32(25)));
            y0_hi = _mm256_add_epi32(y0_hi, _mm256_set1_epi32(128));
            y0_hi = _mm256_srli_epi32(y0_hi, 8);
            y0_hi = _mm256_add_epi32(y0_hi, _mm256_set1_epi32(16));

            // 压缩成 8bit 并存储
            __m256i y0_8 = _mm256_packus_epi16(
                               _mm256_packus_epi32(y0_lo, y0_hi),
                               _mm256_setzero_si256());
            y0_8 = _mm256_permute4x64_epi64(y0_8, 0xD8);
            _mm256_storeu_si256((__m256i*)(dstY0 + i), y0_8);

            // 对下一行再做一次 Y
            r_lo = _mm256_unpacklo_epi16(r1, zero);
            r_hi = _mm256_unpackhi_epi16(r1, zero);
            g_lo = _mm256_unpacklo_epi16(g1, zero);
            g_hi = _mm256_unpackhi_epi16(g1, zero);
            b_lo = _mm256_unpacklo_epi16(b1, zero);
            b_hi = _mm256_unpackhi_epi16(b1, zero);

            __m256i y1_lo = _mm256_add_epi32(_mm256_add_epi32(
                                _mm256_mullo_epi32(r_lo, _mm256_set1_epi32(66)),
                                _mm256_mullo_epi32(g_lo, _mm256_set1_epi32(129))),
                                _mm256_mullo_epi32(b_lo, _mm256_set1_epi32(25)));
            y1_lo = _mm256_add_epi32(y1_lo, _mm256_set1_epi32(128));
            y1_lo = _mm256_srli_epi32(y1_lo, 8);
            y1_lo = _mm256_add_epi32(y1_lo, _mm256_set1_epi32(16));
            __m256i y1_hi = _mm256_add_epi32(_mm256_add_epi32(
                                _mm256_mullo_epi32(r_hi, _mm256_set1_epi32(66)),
                                _mm256_mullo_epi32(g_hi, _mm256_set1_epi32(129))),
                                _mm256_mullo_epi32(b_hi, _mm256_set1_epi32(25)));
            __m256i y1_8 = _mm256_packus_epi16(
                               _mm256_packus_epi32(y1_lo, y1_hi),
                               _mm256_setzero_si256());
            y1_8 = _mm256_permute4x64_epi64(y1_8, 0xD8);
            _mm256_storeu_si256((__m256i*)(dstY1 + i), y1_8);

            // 下采样 U/V：对 2×2 块取平均
            __m256i r2 = _mm256_avg_epu16(r0, r1);
            __m256i g2 = _mm256_avg_epu16(g0, g1);
            __m256i b2 = _mm256_avg_epu16(b0, b1);
            __m256i r2_lo = _mm256_unpacklo_epi16(r2, zero);
            __m256i r2_hi = _mm256_unpackhi_epi16(r2, zero);
            __m256i g2_lo = _mm256_unpacklo_epi16(g2, zero);
            __m256i g2_hi = _mm256_unpackhi_epi16(g2, zero);
            __m256i b2_lo = _mm256_unpacklo_epi16(b2, zero);
            __m256i b2_hi = _mm256_unpackhi_epi16(b2, zero);

            // U = (-38*R - 74*G + 112*B + 128) >> 8 + 128
            __m256i u_lo = _mm256_add_epi32(
                               _mm256_add_epi32(
                                   _mm256_mullo_epi32(r2_lo, _mm256_set1_epi32(-38)),
                                   _mm256_mullo_epi32(g2_lo, _mm256_set1_epi32(-74))),
                               _mm256_mullo_epi32(b2_lo, _mm256_set1_epi32(112)));
            u_lo = _mm256_add_epi32(u_lo, _mm256_set1_epi32(128));
            u_lo = _mm256_srli_epi32(u_lo, 8);
            u_lo = _mm256_add_epi32(u_lo, _mm256_set1_epi32(128));
            __m256i u_hi = _mm256_add_epi32(_mm256_add_epi32(
                                _mm256_mullo_epi32(r2_hi, _mm256_set1_epi32(66)),
                                _mm256_mullo_epi32(g2_hi, _mm256_set1_epi32(129))),
                                _mm256_mullo_epi32(b2_hi, _mm256_set1_epi32(25)));
            __m256i u_8 = _mm256_packus_epi16(
                              _mm256_packus_epi32(u_lo, u_hi),
                              _mm256_setzero_si256());
            u_8 = _mm256_permute4x64_epi64(u_8, 0xD8);
            _mm_storeu_si128((__m128i*)(dstU + (i >> 1)), _mm256_castsi256_si128(u_8));

            // V = (112*R - 94*G - 18*B + 128) >> 8 + 128
            __m256i v_lo = _mm256_add_epi32(
                               _mm256_add_epi32(
                                   _mm256_mullo_epi32(r2_lo, _mm256_set1_epi32(112)),
                                   _mm256_mullo_epi32(g2_lo, _mm256_set1_epi32(-94))),
                               _mm256_mullo_epi32(b2_lo, _mm256_set1_epi32(-18)));
            v_lo = _mm256_add_epi32(v_lo, _mm256_set1_epi32(128));
            v_lo = _mm256_srli_epi32(v_lo, 8);
            v_lo = _mm256_add_epi32(v_lo, _mm256_set1_epi32(128));
            __m256i v_hi =_mm256_add_epi32(
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(r2_hi, _mm256_set1_epi32(112)),
                                _mm256_mullo_epi32(g2_hi, _mm256_set1_epi32(-94))),
                            _mm256_mullo_epi32(b2_hi, _mm256_set1_epi32(-18)));
            v_hi = _mm256_add_epi32(v_hi, _mm256_set1_epi32(128));
            v_hi = _mm256_srli_epi32(v_hi, 8);
            v_hi = _mm256_add_epi32(v_hi, _mm256_set1_epi32(128));
            __m256i v_8 = _mm256_packus_epi16(
                              _mm256_packus_epi32(v_lo, v_hi),
                              _mm256_setzero_si256());
            v_8 = _mm256_permute4x64_epi64(v_8, 0xD8);
            _mm_storeu_si128((__m128i*)(dstV + (i >> 1)), _mm256_castsi256_si128(v_8));

            srcTop += 48;
            srcBot += 48;
        }

        for (; i < w; i += 2) {
            int r0 = srcTop[i*3+0], g0 = srcTop[i*3+1], b0 = srcTop[i*3+2];
            int r1 = srcBot[i*3+0], g1 = srcBot[i*3+1], b1 = srcBot[i*3+2];
            dstY0[i] = (66*r0 + 129*g0 + 25*b0 + 128)>>8 + 16;
            dstY1[i] = (66*r1 + 129*g1 + 25*b1 + 128)>>8 + 16;
            int r2 = (r0+r1)>>1, g2 = (g0+g1)>>1, b2 = (b0+b1)>>1;
            dstU[i>>1] = ((-38*r2 -74*g2 +112*b2 +128)>>8) + 128;
            dstV[i>>1] = ((112*r2 -94*g2 -18*b2 +128)>>8) + 128;
        }
    }
}

#else
#   define HAS_AVX2_RGB24_TO_YUV420 0

void rgb24_to_yuv420_avx2(const uint8_t* rgb, int w, int h, uint8_t* y, uint8_t* u, uint8_t* v) {};

#endif

#include <chrono>

void PutFrame(VideoContext* ctx, iu8* rgbBuffer, i64 width, i64 height) {
    i64 pxCount = width * height;

    if (!ctx->swsCtx) {
        ctx->swsCtx = sws_getContext(
            width, height, AV_PIX_FMT_RGB24,
            width, height, AV_PIX_FMT_YUV420P,
            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
        );
    }

    AVFrame* rgbFrame = av_frame_alloc();
    av_image_alloc(rgbFrame->data, rgbFrame->linesize, width, height, AV_PIX_FMT_RGB24, 1);

    for (int y = 0; y < height; ++y) {
        memcpy(rgbFrame->data[0] + y * rgbFrame->linesize[0], rgbBuffer + y * width * 3, width * 3);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (HAS_AVX2_RGB24_TO_YUV420) rgb24_to_yuv420_avx2(rgbBuffer, width, height, ctx->frame->data[0], ctx->frame->data[1], ctx->frame->data[2]);
    else sws_scale(ctx->swsCtx, (const iu8* const*)rgbFrame->data, rgbFrame->linesize, 0, height, ctx->frame->data, ctx->frame->linesize);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // printf("[PutFrame] sws_scale: %fms\n", ms);

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
