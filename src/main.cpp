#include <stdint.h>
using i64 = long;
using i32 = int;
using i16 = short;
using i8 = char;
using u64 = unsigned long;
using u32 = unsigned int;
using u16 = unsigned short;
using u8 = unsigned char;
using f64 = double;
using f32 = float;

#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <utility>
#include <cstdarg>
#include <filesystem>
#include <zip.h>
#include <nlohmann/json.hpp>
#include <fstream>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswresample/swresample.h>
    #include <libavutil/channel_layout.h>
    #include <libavutil/opt.h>
}

#define CUURENT_VERSION "0.0.1"
#define CHART_META_PATH "meta.json"
#define CHART_META_KEY_CHARTFILE "chart_file"
#define CHART_META_KEY_AUDIOFILE "audio_file"
#define CHART_META_KEY_IMAGEFILE "image_file"
#define CHART_META_KEY_CHARTWASM "chart_wasm"

enum class LogLevel : i32 {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

struct Logger {
    public:
    Logger(LogLevel level) {
        this->level = level;
    }

    void log(LogLevel level, const char* fmt, va_list args) {
        const char* color = _level_to_color(level);
        printf("%s[%s] ", color, _level_to_str(level));
        vprintf(fmt, args);
        printf("\033[0m\n");
    }

    void debug(const char* fmt, ...) { va_list ap; va_start(ap, fmt); log(LogLevel::DEBUG, fmt, ap); va_end(ap); }
    void info(const char* fmt, ...) { va_list ap; va_start(ap, fmt); log(LogLevel::INFO, fmt, ap); va_end(ap); }
    void warn(const char* fmt, ...) { va_list ap; va_start(ap, fmt); log(LogLevel::WARN, fmt, ap); va_end(ap); }
    void error(const char* fmt, ...) { va_list ap; va_start(ap, fmt); log(LogLevel::ERROR, fmt, ap); va_end(ap); }
    void fatal(const char* fmt, ...) { va_list ap; va_start(ap, fmt); log(LogLevel::FATAL, fmt, ap); va_end(ap); }

    private:
    LogLevel level;

    const char* _level_to_str(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARN: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }

    const char* _level_to_color(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "\033[36m";
            case LogLevel::INFO:  return "\033[32m";
            case LogLevel::WARN:  return "\033[33m";
            case LogLevel::ERROR: return "\033[31m";
            case LogLevel::FATAL: return "\033[35m";
            default: return "\033[0m";
        }
    }
};

struct ArgParser {
    public:
    ArgParser(i32 argc, char** argv) {
        this->argc = argc;
        this->argv = argv;
    }

    bool has(const char* key) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], key) == 0) return true;
        }
        return false;
    }

    private:
    i32 argc;
    char** argv;
};

bool file_exists(const char* path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool zip_has(zip_t* zip, const char* name) {
    zip_int64_t index = zip_name_locate(zip, name, 0);
    if (index == -1) {
        return false;
    }
    zip_file_t* file = zip_fopen_index(zip, index, 0);
    if (!file) {
        return false;
    }
    zip_fclose(file);
    return true;
}

nlohmann::json* load_json_from_buffer(u8* buffer, u64 size) {
    if (!buffer || size == 0) return nullptr;
    try {
        auto* j = new nlohmann::json;
        *j = nlohmann::json::parse((const u8*)buffer, (const u8*)(buffer + size));
        return j;
    } catch (...) {
        return nullptr;
    }
}

nlohmann::json* load_json_from_path(const char* path) {
    if (!path) return nullptr;
    try {
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs) return nullptr;
        auto end = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        auto size = (u64)(end - ifs.tellg());
        if (size == 0) return nullptr;
        std::vector<u8> buf(size);
        if (!ifs.read((i8*)(buf.data()), size)) return nullptr;
        return load_json_from_buffer(buf.data(), size);
    } catch (...) {
        return nullptr;
    }
}

nlohmann::json* load_json_from_zip(zip_t* zip, const char* name) {
    if (!zip || !name) return nullptr;
    zip_int64_t index = zip_name_locate(zip, name, 0);
    if (index == -1) return nullptr;
    zip_file_t* file = zip_fopen_index(zip, index, 0);
    if (!file) return nullptr;
    zip_stat_t stat;
    if (zip_stat_index(zip, index, 0, &stat) != 0) {
        zip_fclose(file);
        return nullptr;
    }
    std::vector<u8> buf(stat.size);
    if (zip_fread(file, buf.data(), stat.size) != stat.size) {
        zip_fclose(file);
        return nullptr;
    }
    zip_fclose(file);
    return load_json_from_buffer(buf.data(), stat.size);
}

std::vector<u8> read_file_from_zip(zip_t* zip, const char* name) {
    if (!zip || !name) return {};
    zip_int64_t index = zip_name_locate(zip, name, 0);
    if (index == -1) return {};
    zip_file_t* file = zip_fopen_index(zip, index, 0);
    if (!file) return {};
    zip_stat_t stat;
    if (zip_stat_index(zip, index, 0, &stat) != 0) {
        zip_fclose(file);
        return {};
    }
    std::vector<u8> buf(stat.size);
    if (zip_fread(file, buf.data(), stat.size) != stat.size) {
        zip_fclose(file);
        return {};
    }
    zip_fclose(file);
    return buf;
}

const char* json_read_string_key(nlohmann::json* json, const char* key) noexcept {
    if (!json || !json->is_object()) return nullptr;
    auto it = json->find(key);
    if (it == json->end() || !it->is_string()) return nullptr;
    auto sp = it->get_ptr<const std::string*>();
    return sp->c_str();
}

nlohmann::json* json_read_object_key(nlohmann::json* json, const char* key) noexcept {
    if (!json || !json->is_object()) return nullptr;
    auto it = json->find(key);
    if (it == json->end() || !it->is_object()) return nullptr;
    return &(*it);
}

nlohmann::json* json_read_array_key(nlohmann::json* json, const char* key) noexcept {
    if (!json || !json->is_object()) return nullptr;
    auto it = json->find(key);
    if (it == json->end() || !it->is_array()) return nullptr;
    return &(*it);
}

nlohmann::json* json_read_number_key(nlohmann::json* json, const char* key) noexcept {
    if (!json || !json->is_object()) return nullptr;
    auto it = json->find(key);
    if (it == json->end() || !it->is_number()) return nullptr;
    return it->get_ptr<nlohmann::json*>();
}

std::vector<u8> decode_audio_pcm16le44100(const std::vector<u8>& data) {
    if (data.empty()) return {};

    constexpr int IO_BUF_SIZE = 64 * 1024;
    u8* io_buf = (u8*)av_malloc(IO_BUF_SIZE);
    if (!io_buf) return {};

    struct IOCtx { const u8* cur; const u8* end; } io_ctx{ data.data(), data.data() + data.size() };

    auto read_fn = +[](void* opaque, uint8_t* buf, int buf_size) -> int
    {
        auto& io = *static_cast<IOCtx*>(opaque);
        int left = int(io.end - io.cur);
        int n = std::min(left, buf_size);
        if (n <= 0) return AVERROR_EOF;
        memcpy(buf, io.cur, n);
        io.cur += n;
        return n;
    };

    AVIOContext* avio = avio_alloc_context(io_buf, IO_BUF_SIZE, 0, &io_ctx, read_fn, nullptr, nullptr);
    if (!avio) { av_free(io_buf); return {}; }

    AVFormatContext* fmt = avformat_alloc_context();
    fmt->pb = avio;
    if (avformat_open_input(&fmt, nullptr, nullptr, nullptr) < 0) {
        avformat_free_context(fmt);
        av_free(io_buf);
        return {};
    }

    auto fail = [&]() {
        avformat_close_input(&fmt);
        av_free(io_buf);
        return std::vector<u8>{};
    };

    if (avformat_find_stream_info(fmt, nullptr) < 0) return fail();

    int stream_idx = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_idx < 0) return fail();
    AVCodecParameters* par = fmt->streams[stream_idx]->codecpar;

    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    if (!codec) return fail();
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    if (!dec) return fail();
    auto dec_deleter = [](AVCodecContext* c){ avcodec_free_context(&c); };
    std::unique_ptr<AVCodecContext, decltype(dec_deleter)> dec_guard(dec, dec_deleter);

    if (avcodec_parameters_to_context(dec, par) < 0 ||
        avcodec_open2(dec, codec, nullptr) < 0) return fail();

    SwrContext* swr = swr_alloc();
    if (!swr) return fail();
    auto swr_deleter = [](SwrContext* s){ swr_free(&s); };
    std::unique_ptr<SwrContext, decltype(swr_deleter)> swr_guard(swr, swr_deleter);

    av_opt_set_int(swr, "in_channel_count",  dec->ch_layout.nb_channels, 0);
    av_opt_set_int(swr, "in_sample_rate",    dec->sample_rate, 0);
    av_opt_set_sample_fmt(swr, "in_sample_fmt",  dec->sample_fmt, 0);

    AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
    av_opt_set_chlayout(swr, "out_chlayout", &stereo, 0);
    av_opt_set_int(swr, "out_sample_rate", 48000, 0);
    av_opt_set_sample_fmt(swr, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
    if (swr_init(swr) < 0) return fail();

    AVPacket* pkt  = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!pkt || !frame) return fail();
    auto pkt_deleter   = [](AVPacket* p){   av_packet_free(&p); };
    auto frame_deleter = [](AVFrame* f){    av_frame_free(&f); };
    std::unique_ptr<AVPacket, decltype(pkt_deleter)>   pkt_guard(pkt, pkt_deleter);
    std::unique_ptr<AVFrame, decltype(frame_deleter)> frame_guard(frame, frame_deleter);

    std::vector<u8> pcm;
    auto convert_frame = [&]()
    {
        int out_samples = av_rescale_rnd(swr_get_delay(swr, frame->sample_rate) + frame->nb_samples, 48000, frame->sample_rate, AV_ROUND_UP);
        std::vector<u8> conv(out_samples * 2 * 2); // s16 * stereo
        u8* out_ptr = conv.data();
        int got = swr_convert(swr, &out_ptr, out_samples, (const u8**)frame->data, frame->nb_samples);
        if (got > 0) pcm.insert(pcm.end(), conv.begin(), conv.begin() + got * 4);
    };

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != stream_idx) { av_packet_unref(pkt); continue; }
        if (avcodec_send_packet(dec, pkt) == 0)
            while (avcodec_receive_frame(dec, frame) == 0)
                convert_frame();
        av_packet_unref(pkt);
    }

    avcodec_send_packet(dec, nullptr);
    while (avcodec_receive_frame(dec, frame) == 0)
        convert_frame();

    return pcm;
}

std::vector<u8> load_file_as_vector(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    if (size == 0) return {};
    std::vector<u8> buf(size);
    ifs.seekg(0, std::ios::beg);
    ifs.read((i8*)buf.data(), size);
    return buf;
}

std::vector<u8> load_audio_from_path(const char* path) {
    return decode_audio_pcm16le44100(load_file_as_vector(path));
}

i32 main(i32 argc, char** argv) {
    Logger logger = Logger(LogLevel::DEBUG);
    logger.info("mil renderer v%s", CUURENT_VERSION);

    ArgParser arg_parser = ArgParser(argc, argv);

    if (argc < 3 || arg_parser.has("-h") || arg_parser.has("--help")) {
        logger.info("Usage: <input chart file> <output video file>");
        return (i32)(argc < 3);
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    if (!file_exists(input_file)) {
        logger.error("input file does not exist: %s", input_file);
        return 1;
    }

    if (file_exists(output_file)) {
        logger.error("output file already exists: %s", output_file);
        return 1;
    }

    logger.debug("input file: %s", input_file);
    logger.debug("output file: %s", output_file);

    zip_t *zip = zip_open(input_file, ZIP_RDONLY, NULL);
    if (!zip) {
        logger.error("failed to open zip file: %s", input_file);
        if (load_json_from_path(input_file) != nullptr) {
            logger.info("input file is a json file, please export as zip in editor");
        }
        return 1;
    }

    logger.debug("zip file opened: %p", zip);

    if (!zip_has(zip, CHART_META_PATH)) {
        logger.fatal("zip file does not have %s, invalid milthm chart file", CHART_META_PATH);
        return 1;
    }

    nlohmann::json* meta_json = load_json_from_zip(zip, CHART_META_PATH);
    if (meta_json == nullptr) {
        logger.fatal("failed to load meta.json from zip file");
        return 1;
    }

    logger.debug("meta.json loaded: %p", meta_json);
    const char* chart_file_path = json_read_string_key(meta_json, CHART_META_KEY_CHARTFILE);
    const char* audio_file_path = json_read_string_key(meta_json, CHART_META_KEY_AUDIOFILE);
    const char* image_file_path = json_read_string_key(meta_json, CHART_META_KEY_IMAGEFILE);

    if (chart_file_path == nullptr || audio_file_path == nullptr || image_file_path == nullptr) {
        logger.fatal("meta.json does not have all required keys or some keys are not strings");
        return 1;
    }

    logger.debug("chart file path: %s", chart_file_path);
    logger.debug("audio file path: %s", audio_file_path);
    logger.debug("image file path: %s", image_file_path);

    if (!zip_has(zip, chart_file_path)) {
        logger.fatal("zip file does not have %s", chart_file_path);
        return 1;
    }
    
    if (!zip_has(zip, audio_file_path)) {
        logger.fatal("zip file does not have %s", audio_file_path);
        return 1;
    }

    if (!zip_has(zip, image_file_path)) {
        logger.fatal("zip file does not have %s", image_file_path);
        return 1;
    }

    nlohmann::json* chart_json = load_json_from_zip(zip, chart_file_path);
    if (chart_json == nullptr) {
        logger.fatal("failed to load %s as json from zip file", chart_file_path);
        return 1;
    }

    logger.debug("chart json loaded: %p", chart_json);

    std::vector<u8> audio_data = read_file_from_zip(zip, audio_file_path);
    if (audio_data.empty()) {
        logger.fatal("failed to read %s from zip file", audio_file_path);
        return 1;
    }

    logger.debug("audio data loaded: %p", (void*)audio_data.data());

    std::vector<u8> auduo_pcm = decode_audio_pcm16le44100(audio_data);
    logger.debug("audio pcm decoded: %p", (void*)auduo_pcm.data());

    std::vector<u8> hitsound_hit = load_audio_from_path("./res/hit.ogg");
    std::vector<u8> hitsound_drag = load_audio_from_path("./res/drag.ogg");

    if (hitsound_hit.empty() || hitsound_drag.empty()) {
        logger.fatal("failed to load hitsound audio");
        return 1;
    }

    logger.debug("hitsound audio loaded (hit): %p", (void*)hitsound_hit.data());
    logger.debug("hitsound audio loaded (drag): %p", (void*)hitsound_drag.data());

    nlohmann::json* jlines = json_read_array_key(chart_json, "lines");
    if (jlines == nullptr) {
        logger.fatal("chart json does not have lines key or it is not an array");
        return 1;
    }

    // for (auto jline : *jlines) {
    //     if (!jline->is_object()) {
    //         logger.fatal("line is not an object");
    //         return 1;
    //     }

    //     nlohmann::json* jnotes = json_read_array_key(jline, "notes");
    //     if (jnotes == nullptr) {
    //         logger.fatal("line does not have notes key or it is not an array");
    //         return 1;
    //     }

    //     for (auto jnote : *jnotes) {
    //         if (!jnote.is_object()) {
    //             logger.fatal("note is not an object");
    //             return 1;
    //         }

    //         nlohmann::json* jtype = json_read_number_key(jnote, "type");
    //         if (jtype == nullptr) {
    //             logger.fatal("note does not have type key or it is not a number");
    //             return 1;
    //         }

    //         f64 type = jtype->get<f64>();
    //         logger.debug("note type: %f", type);
    //     }
    // }

    return 0;
}

狗屎
