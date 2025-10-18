#include <stdint.h>
#define i64 long
#define i32 int
#define i16 short
#define i8 char
#define u64 unsigned i64
#define u32 unsigned i32
#define u16 unsigned i16
#define u8 unsigned i8
#define f64 double
#define f32 float
#define uf64 unsigned f64
#define uf32 unsigned f32

#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <utility>
#include <cstdarg>
#include <filesystem>
#include <zip.h>
#include <nlohmann/json.hpp>
#include <fstream>

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

    logger.debug("audio data loaded: %p", audio_data);

    return 0;
}
