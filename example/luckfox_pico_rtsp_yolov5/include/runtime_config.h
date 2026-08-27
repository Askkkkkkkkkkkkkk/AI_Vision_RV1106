#ifndef RUNTIME_CONFIG_H
#define RUNTIME_CONFIG_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct RuntimeConfig {
    char model_path[256];
    char label_path[256];
    float box_thresh;
    float nms_thresh;
};

inline RuntimeConfig& runtime_config()
{
    static RuntimeConfig config = {
        "./model/yolov5.rknn",
        "./model/coco_80_labels_list.txt",
        0.25f,
        0.45f
    };
    return config;
}

inline char* trim_config_text(char* text)
{
    while (*text && isspace((unsigned char)*text)) {
        ++text;
    }

    char* end = text + strlen(text);
    while (end > text && isspace((unsigned char)*(end - 1))) {
        --end;
    }
    *end = '\0';
    return text;
}

inline void set_config_string(char* destination, size_t size, const char* value)
{
    snprintf(destination, size, "%s", value);
}

inline bool load_runtime_config(const char* path)
{
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        printf("[CONFIG] Cannot open %s, using defaults\n", path);
        return false;
    }

    RuntimeConfig& config = runtime_config();
    char line[512];

    while (fgets(line, sizeof(line), file) != NULL) {
        char* text = trim_config_text(line);
        if (*text == '\0' || *text == '#') {
            continue;
        }

        char* equal = strchr(text, '=');
        if (equal == NULL) {
            continue;
        }

        *equal = '\0';
        char* key = trim_config_text(text);
        char* value = trim_config_text(equal + 1);

        if (strcmp(key, "model_path") == 0) {
            set_config_string(config.model_path, sizeof(config.model_path), value);
        } else if (strcmp(key, "label_path") == 0) {
            set_config_string(config.label_path, sizeof(config.label_path), value);
        } else if (strcmp(key, "box_thresh") == 0) {
            config.box_thresh = strtof(value, NULL);
        } else if (strcmp(key, "nms_thresh") == 0) {
            config.nms_thresh = strtof(value, NULL);
        }
    }

    fclose(file);

    printf("[CONFIG] model_path=%s\n", config.model_path);
    printf("[CONFIG] label_path=%s\n", config.label_path);
    printf("[CONFIG] box_thresh=%.2f, nms_thresh=%.2f\n",
           config.box_thresh, config.nms_thresh);
    return true;
}

#endif
