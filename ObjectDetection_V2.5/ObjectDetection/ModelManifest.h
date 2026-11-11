#pragma once

#include <map>
#include <string>
#include <vector>

#include "InferenceTypes.h"

struct ModelInputSpec
{
    int width = 0;
    int height = 0;
    int channels = 3;
    std::string format = "RGB";
    TensorLayout layout = TensorLayout::Unknown;
    TensorDataType type = TensorDataType::Unknown;

    bool valid() const
    {
        return width > 0 && height > 0 && channels > 0 &&
               layout != TensorLayout::Unknown &&
               type != TensorDataType::Unknown;
    }
};

struct ModelPreprocessSpec
{
    std::string type = "letterbox";
    bool keep_aspect_ratio = true;
    std::vector<float> mean{0.0f, 0.0f, 0.0f};
    std::vector<float> std{255.0f, 255.0f, 255.0f};
    std::vector<int> padding_value{114, 114, 114};
};

struct ModelOutputSpec
{
    std::string type;
    int num_classes = 0;
    int reg_max = 0;
    std::vector<int> strides;
    bool class_scores_activated = false;
};

struct ModelArtifactProfile
{
    std::string profile_id;
    std::string backend;
    std::string target_platform;
    std::map<std::string, std::string> variants;
    std::string default_variant;
    bool has_input_override = false;
    ModelInputSpec input;
};

struct ResolvedModelArtifact
{
    std::string profile_id;
    std::string backend;
    std::string target_platform;
    std::string variant;
    std::string relative_path;
    ModelInputSpec input;
};

struct ModelManifest
{
    int schema_version = 2;
    std::string model_id;
    std::string model_name;
    std::string model_version;
    std::string task_type;
    std::string object_type;
    std::string adapter;
    std::map<std::string, ModelArtifactProfile> artifact_profiles;
    ModelInputSpec input;
    ModelPreprocessSpec preprocess;
    ModelOutputSpec output;
    std::string labels_file = "labels.txt";

    bool resolveArtifact(const std::string& platform_profile,
                         const std::string& compiled_backend,
                         const std::string& requested_variant,
                         ResolvedModelArtifact& resolved,
                         std::string& error) const;
};

class ModelManifestLoader
{
public:
    static bool load(const std::string& path,
                     ModelManifest& manifest,
                     std::string& error);

    static std::string normalizePlatform(const std::string& value);
    static bool platformCompatible(const std::string& requested,
                                   const std::string& declared);

private:
    static std::string lower(const std::string& value);
};
