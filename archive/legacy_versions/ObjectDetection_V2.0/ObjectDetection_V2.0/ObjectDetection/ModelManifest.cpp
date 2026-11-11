#include "ModelManifest.h"

#include <algorithm>
#include <cctype>

#include <opencv2/core.hpp>

namespace
{
std::string readString(const cv::FileNode& node,
                       const std::string& fallback = std::string())
{
    if (node.empty() || !node.isString()) return fallback;
    return static_cast<std::string>(node);
}

int readInt(const cv::FileNode& node, int fallback)
{
    if (node.empty() || (!node.isInt() && !node.isReal())) return fallback;
    return static_cast<int>(node);
}

bool readBool(const cv::FileNode& node, bool fallback)
{
    if (node.empty()) return fallback;
    if (node.isInt()) return static_cast<int>(node) != 0;
    if (!node.isString()) return fallback;

    std::string value = static_cast<std::string>(node);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    return fallback;
}

std::vector<float> readFloatArray(const cv::FileNode& node,
                                  const std::vector<float>& fallback)
{
    if (node.empty() || !node.isSeq()) return fallback;
    std::vector<float> values;
    for (auto it = node.begin(); it != node.end(); ++it)
    {
        if ((*it).isInt() || (*it).isReal())
            values.push_back(static_cast<float>(static_cast<double>(*it)));
    }
    return values.empty() ? fallback : values;
}

std::vector<int> readIntArray(const cv::FileNode& node,
                              const std::vector<int>& fallback)
{
    if (node.empty() || !node.isSeq()) return fallback;
    std::vector<int> values;
    for (auto it = node.begin(); it != node.end(); ++it)
    {
        if ((*it).isInt() || (*it).isReal())
            values.push_back(static_cast<int>(*it));
    }
    return values.empty() ? fallback : values;
}

std::string lowerCopy(const std::string& value)
{
    std::string output = value;
    std::transform(output.begin(), output.end(), output.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return output;
}

TensorLayout parseLayout(const std::string& value)
{
    const std::string id = lowerCopy(value);
    if (id == "nhwc") return TensorLayout::NHWC;
    if (id == "nchw") return TensorLayout::NCHW;
    return TensorLayout::Unknown;
}

TensorDataType parseType(const std::string& value)
{
    const std::string id = lowerCopy(value);
    if (id == "uint8" || id == "u8") return TensorDataType::UInt8;
    if (id == "float" || id == "float32" || id == "fp32")
        return TensorDataType::Float32;
    return TensorDataType::Unknown;
}

ModelInputSpec readInputSpec(const cv::FileNode& node,
                             const ModelInputSpec& fallback)
{
    if (node.empty() || !node.isMap()) return fallback;

    ModelInputSpec input = fallback;
    input.width = readInt(node["width"], input.width);
    input.height = readInt(node["height"], input.height);
    input.channels = readInt(node["channels"], input.channels);
    input.format = readString(node["format"], input.format);

    const std::string layout = readString(node["layout"]);
    if (!layout.empty()) input.layout = parseLayout(layout);

    const std::string type = readString(node["type"]);
    if (!type.empty()) input.type = parseType(type);

    return input;
}

void readVariants(const cv::FileNode& node,
                  std::map<std::string, std::string>& variants)
{
    if (node.empty() || !node.isMap()) return;
    for (auto it = node.begin(); it != node.end(); ++it)
    {
        if ((*it).isString())
            variants[lowerCopy((*it).name())] = readString(*it);
    }
}

bool resolveVariant(const ModelArtifactProfile& profile,
                    const std::string& requested_variant,
                    std::string& resolved_variant,
                    std::string& relative_path)
{
    if (profile.variants.empty()) return false;

    std::string variant = lowerCopy(requested_variant);
    if (variant.empty() || variant == "default")
        variant = lowerCopy(profile.default_variant);

    auto it = profile.variants.find(variant);
    if (it == profile.variants.end())
        it = profile.variants.find(lowerCopy(profile.default_variant));
    if (it == profile.variants.end())
        it = profile.variants.begin();

    resolved_variant = it->first;
    relative_path = it->second;
    return true;
}
}

bool ModelManifest::resolveArtifact(const std::string& platform_profile,
                                    const std::string& compiled_backend,
                                    const std::string& requested_variant,
                                    ResolvedModelArtifact& resolved,
                                    std::string& error) const
{
    resolved = ResolvedModelArtifact{};
    error.clear();

    const std::string requested_profile = lowerCopy(platform_profile);
    const std::string backend = lowerCopy(compiled_backend);
    const ModelArtifactProfile* selected = nullptr;

    auto exact = artifact_profiles.find(requested_profile);
    if (exact != artifact_profiles.end()) selected = &exact->second;

    if (!selected)
    {
        for (const auto& item : artifact_profiles)
        {
            const ModelArtifactProfile& candidate = item.second;
            const std::string declared = candidate.target_platform.empty()
                ? candidate.profile_id
                : candidate.target_platform;
            if (ModelManifestLoader::platformCompatible(requested_profile, declared))
            {
                selected = &candidate;
                break;
            }
        }
    }

    if (!selected)
    {
        error = "platform profile not found in model package: " + platform_profile;
        return false;
    }

    if (lowerCopy(selected->backend) != backend)
    {
        error = "artifact backend=" + selected->backend +
                " does not match compiled backend=" + compiled_backend;
        return false;
    }

    if (!resolveVariant(*selected,
                        requested_variant,
                        resolved.variant,
                        resolved.relative_path))
    {
        error = "no model variants for platform_profile=" + platform_profile;
        return false;
    }

    resolved.profile_id = selected->profile_id;
    resolved.backend = lowerCopy(selected->backend);
    resolved.target_platform = lowerCopy(selected->target_platform);
    resolved.input = selected->has_input_override ? selected->input : input;

    if (!resolved.input.valid())
    {
        error = "invalid input contract for platform_profile=" + platform_profile;
        return false;
    }

    return true;
}

bool ModelManifestLoader::load(const std::string& path,
                               ModelManifest& manifest,
                               std::string& error)
{
    manifest = ModelManifest{};
    error.clear();

    cv::FileStorage fs;
    try
    {
        if (!fs.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON))
        {
            error = "cannot open model manifest: " + path;
            return false;
        }
    }
    catch (const cv::Exception& e)
    {
        error = std::string("invalid model manifest: ") + e.what();
        return false;
    }

    const cv::FileNode root = fs.root();
    manifest.schema_version = readInt(root["schema_version"], 0);
    if (manifest.schema_version < 2)
    {
        error = "model manifest schema_version must be >= 2";
        return false;
    }

    manifest.model_id = readString(root["model_id"]);
    manifest.model_name = readString(root["model_name"], manifest.model_id);
    manifest.model_version = readString(root["version"]);
    manifest.task_type = lower(readString(root["task_type"]));
    manifest.object_type = lower(readString(root["object_type"]));
    manifest.adapter = lower(readString(root["adapter"]));
    manifest.input = readInputSpec(root["input"], manifest.input);

    const cv::FileNode preprocess = root["preprocess"];
    manifest.preprocess.type = lower(readString(preprocess["type"], "letterbox"));
    manifest.preprocess.keep_aspect_ratio =
        readBool(preprocess["keep_aspect_ratio"], true);
    manifest.preprocess.mean =
        readFloatArray(preprocess["mean"], manifest.preprocess.mean);
    manifest.preprocess.std =
        readFloatArray(preprocess["std"], manifest.preprocess.std);

    if (!preprocess["padding_value"].empty())
    {
        if (preprocess["padding_value"].isSeq())
        {
            manifest.preprocess.padding_value =
                readIntArray(preprocess["padding_value"],
                             manifest.preprocess.padding_value);
        }
        else
        {
            const int value = readInt(preprocess["padding_value"], 114);
            manifest.preprocess.padding_value = {value, value, value};
        }
    }

    const cv::FileNode output = root["output"];
    manifest.output.type = lower(readString(output["type"]));
    manifest.output.num_classes = readInt(output["num_classes"], 0);
    manifest.output.reg_max = readInt(output["reg_max"], 0);
    manifest.output.strides =
        readIntArray(output["strides"], manifest.output.strides);
    manifest.output.class_scores_activated =
        readBool(output["class_scores_activated"], false);

    manifest.labels_file = readString(root["labels"], "labels.txt");

    const cv::FileNode profiles = root["artifacts"]["profiles"];
    if (!profiles.empty() && profiles.isMap())
    {
        for (auto it = profiles.begin(); it != profiles.end(); ++it)
        {
            ModelArtifactProfile profile;
            profile.profile_id = lower((*it).name());
            profile.backend = lower(readString((*it)["backend"]));
            profile.target_platform =
                lower(readString((*it)["target_platform"], profile.profile_id));
            profile.default_variant = lower(readString((*it)["default"]));
            readVariants((*it)["variants"], profile.variants);

            if (profile.default_variant.empty() && !profile.variants.empty())
                profile.default_variant = profile.variants.begin()->first;

            const cv::FileNode profile_input = (*it)["input"];
            if (!profile_input.empty())
            {
                profile.input = readInputSpec(profile_input, manifest.input);
                profile.has_input_override = true;
            }

            manifest.artifact_profiles[profile.profile_id] = profile;
        }
    }

    if (manifest.model_id.empty())
    {
        error = "model_id is missing";
        return false;
    }
    if (manifest.model_version.empty())
    {
        error = "version is missing";
        return false;
    }
    if (manifest.task_type != "bbox")
    {
        error = "unsupported task_type: " + manifest.task_type;
        return false;
    }
    if (manifest.object_type.empty())
    {
        error = "object_type is missing";
        return false;
    }
    if (manifest.adapter.empty())
    {
        error = "adapter is missing";
        return false;
    }
    if (manifest.output.type.empty())
    {
        error = "output.type is missing";
        return false;
    }
    if (manifest.output.type != manifest.adapter)
    {
        error = "adapter and output.type must match";
        return false;
    }
    if (manifest.output.num_classes <= 0)
    {
        error = "output.num_classes must be positive";
        return false;
    }

    if (manifest.adapter == "dfl_multiscale_class_scores_v1")
    {
        if (manifest.output.reg_max <= 1)
        {
            error = "DFL output reg_max must be greater than 1";
            return false;
        }
        if (manifest.output.strides.empty())
        {
            error = "DFL output strides are missing";
            return false;
        }
        if (!manifest.output.class_scores_activated)
        {
            error = "DFL output requires class_scores_activated=true";
            return false;
        }
    }

    if (manifest.artifact_profiles.empty())
    {
        error = "artifacts.profiles is missing";
        return false;
    }

    for (const auto& item : manifest.artifact_profiles)
    {
        const ModelArtifactProfile& profile = item.second;
        if (profile.backend.empty())
        {
            error = "artifact profile backend is missing: " + item.first;
            return false;
        }
        if (profile.variants.empty())
        {
            error = "artifact profile variants are missing: " + item.first;
            return false;
        }
        const ModelInputSpec selected_input =
            profile.has_input_override ? profile.input : manifest.input;
        if (!selected_input.valid())
        {
            error = "artifact profile input is invalid: " + item.first;
            return false;
        }
    }

    return true;
}

std::string ModelManifestLoader::normalizePlatform(const std::string& value)
{
    const std::string platform = lower(value);

    if (platform == "orangepi5b" ||
        platform == "orangepi5b_rk3588" ||
        platform == "rk3588")
    {
        return "rk3588";
    }

    if (platform == "jetson" ||
        platform == "jetson_orin" ||
        platform == "jetson_orin_nano")
    {
        return "jetson_orin_nano";
    }

    return platform;
}

bool ModelManifestLoader::platformCompatible(const std::string& requested,
                                             const std::string& declared)
{
    if (requested.empty() || declared.empty()) return true;
    return normalizePlatform(requested) == normalizePlatform(declared);
}

std::string ModelManifestLoader::lower(const std::string& value)
{
    return lowerCopy(value);
}
