#include "common/StringParsers.h"
#include "common/Base64.h"

#include <rapidjson/document.h>
#include <cstring>
#include <algorithm>

namespace
{
    using namespace rapidjson;

    bool readUint64(const Value& obj, const char* key, uint64_t& out)
    {
        if (!obj.HasMember(key)) return false;
        if (obj[key].IsUint64()) {
            out = obj[key].GetUint64();
            return true;
        }
        if (obj[key].IsInt64()) {
            out = static_cast<uint64_t>(obj[key].GetInt64());
            return true;
        }
        return false;
    }

    bool readUint32(const Value& obj, const char* key, uint32_t& out)
    {
        if (!obj.HasMember(key)) return false;
        if (obj[key].IsUint()) {
            out = obj[key].GetUint();
            return true;
        }
        if (obj[key].IsInt()) {
            int v = obj[key].GetInt();
            if (v < 0) v = 0;
            out = static_cast<uint32_t>(v);
            return true;
        }
        return false;
    }

    bool readInt32(const Value& obj, const char* key, int32_t& out)
    {
        if (!obj.HasMember(key) || !obj[key].IsInt()) return false;
        out = obj[key].GetInt();
        return true;
    }

    bool readFloat(const Value& obj, const char* key, float& out)
    {
        if (!obj.HasMember(key)) return false;
        if (!obj[key].IsNumber()) return false;
        out = static_cast<float>(obj[key].GetDouble());
        return true;
    }

    bool readDouble(const Value& obj, const char* key, double& out)
    {
        if (!obj.HasMember(key) || !obj[key].IsNumber()) return false;
        out = obj[key].GetDouble();
        return true;
    }

    void parseHeader(const rapidjson::Value& root, tmv::Header& h)
    {
        if (!root.HasMember("header") || !root["header"].IsObject()) return;
        const auto& hdr = root["header"];
        readUint64(hdr, "pub_nsec", h.pub_nsec);
        readUint64(hdr, "data_nsec", h.data_nsec);
    }

    void parseIntrinsics(const rapidjson::Value& root, tmv::Intrinsics& intr)
    {
        if (!root.HasMember("intr") || !root["intr"].IsObject()) return;
        const auto& v = root["intr"];

        readUint32(v, "width", intr.width);
        readUint32(v, "height", intr.height);
        readFloat(v, "fx", intr.fx);
        readFloat(v, "fy", intr.fy);
        readFloat(v, "ppx", intr.ppx);
        readFloat(v, "ppy", intr.ppy);
        readInt32(v, "model", intr.model);

        intr.coeffs.clear();
        if (v.HasMember("coeffs") && v["coeffs"].IsArray()) {
            for (const auto& c : v["coeffs"].GetArray()) {
                if (c.IsNumber()) {
                    intr.coeffs.push_back(static_cast<float>(c.GetDouble()));
                }
            }
        }
    }

    bool readBoolAny(const rapidjson::Value& obj, const char* key1, const char* key2, bool& out)
    {
        for (const char* key : {key1, key2}) {
            if (obj.HasMember(key) && obj[key].IsBool()) {
                out = obj[key].GetBool();
                return true;
            }
        }
        return false;
    }

    bool parseFloatArray(const rapidjson::Value& value, float* dst, size_t count)
    {
        if (!value.IsArray() || value.GetArray().Size() != count) return false;
        for (rapidjson::SizeType i = 0; i < value.GetArray().Size(); ++i) {
            if (!value[i].IsNumber()) return false;
            dst[i] = static_cast<float>(value[i].GetDouble());
        }
        return true;
    }

    void parseDepthToColorCalibration(const rapidjson::Value& root, tmv::DepthFrame& out)
    {
        readBoolAny(root, "registered_to_color", "depth_registered_to_color", out.registered_to_color);
        readBoolAny(root, "registeredToColor", "depthRegisteredToColor", out.registered_to_color);

        const rapidjson::Value* ext = nullptr;
        for (const char* key : {"depth_to_color_extrinsics", "depthToColorExtrinsics", "depth_to_color"}) {
            if (root.HasMember(key) && root[key].IsObject()) {
                ext = &root[key];
                break;
            }
        }
        if (!ext) return;

        const rapidjson::Value* rotation = nullptr;
        const rapidjson::Value* translation = nullptr;
        for (const char* key : {"rotation", "R", "rot"})
            if (ext->HasMember(key)) { rotation = &(*ext)[key]; break; }
        for (const char* key : {"translation", "t", "trans"})
            if (ext->HasMember(key)) { translation = &(*ext)[key]; break; }

        out.has_depth_to_color_extrinsics = rotation && translation &&
            parseFloatArray(*rotation, out.depth_to_color_rotation.data(), 9) &&
            parseFloatArray(*translation, out.depth_to_color_translation.data(), 3);
    }

    bool parsePixelUV(const rapidjson::Value& obj, tmv::PixelUV& px)
    {
        if (!obj.IsObject()) return false;
        bool okU = readUint32(obj, "u", px.u);
        bool okV = readUint32(obj, "v", px.v);
        return okU && okV;
    }

    bool decodeBytesToU8(const rapidjson::Value& root, const char* key, std::vector<uint8_t>& out)
    {
        if (!root.HasMember(key) || !root[key].IsString()) return false;
        return Base64::decode(root[key].GetString(), out);
    }

    bool decodeBytesToU16(const rapidjson::Value& root, const char* key, std::vector<uint16_t>& out)
    {
        std::vector<uint8_t> raw;
        if (!decodeBytesToU8(root, key, raw)) {
            return false;
        }

        if (raw.size() % sizeof(uint16_t) != 0) {
            return false;
        }

        out.resize(raw.size() / sizeof(uint16_t));
        std::memcpy(out.data(), raw.data(), raw.size());
        return true;
    }
}

bool readUint32Any(const rapidjson::Value& obj,
                   const char* key1,
                   const char* key2,
                   uint32_t& out)
{
    if (obj.HasMember(key1)) {
        return readUint32(obj, key1, out);
    }
    if (obj.HasMember(key2)) {
        return readUint32(obj, key2, out);
    }
    return false;
}

bool readFloatAny(const rapidjson::Value& obj,
                  const char* key1,
                  const char* key2,
                  float& out)
{
    if (obj.HasMember(key1)) {
        return readFloat(obj, key1, out);
    }
    if (obj.HasMember(key2)) {
        return readFloat(obj, key2, out);
    }
    return false;
}

bool readDoubleAny(const rapidjson::Value& obj,
                   const char* key1,
                   const char* key2,
                   double& out)
{
    if (obj.HasMember(key1)) {
        return readDouble(obj, key1, out);
    }
    if (obj.HasMember(key2)) {
        return readDouble(obj, key2, out);
    }
    return false;
}

bool decodeBytesToU8Any(const rapidjson::Value& root,
                        const char* key1,
                        const char* key2,
                        std::vector<uint8_t>& out)
{
    if (root.HasMember(key1)) {
        return decodeBytesToU8(root, key1, out);
    }
    if (root.HasMember(key2)) {
        return decodeBytesToU8(root, key2, out);
    }
    return false;
}

bool decodeBytesToU16Any(const rapidjson::Value& root,
                         const char* key1,
                         const char* key2,
                         std::vector<uint16_t>& out)
{
    if (root.HasMember(key1)) {
        return decodeBytesToU16(root, key1, out);
    }
    if (root.HasMember(key2)) {
        return decodeBytesToU16(root, key2, out);
    }
    return false;
}

bool StringParsers::parseImageFrame(const std::string& data, tmv::ImageFrame& out)
{
    rapidjson::Document doc;
    if (doc.Parse(data.c_str()).HasParseError() || !doc.IsObject()) {
        return false;
    }

    out = tmv::ImageFrame{};
    parseHeader(doc, out.header);
    parseIntrinsics(doc, out.intr);

    readUint32(doc, "width", out.width);
    readUint32(doc, "height", out.height);

    readUint32Any(doc, "stride_bytes", "strideBytes", out.stride_bytes);
    readDoubleAny(doc, "rs_timestamp_ms", "rsTimestampMs", out.rs_timestamp_ms);

    if (!decodeBytesToU8Any(doc, "bgr8", "bgr8", out.bgr8)) {
        return false;
    }

    if (out.stride_bytes == 0 && out.width > 0) {
        out.stride_bytes = out.width * 3;
    }

    out.valid = (out.width > 0 && out.height > 0 && !out.bgr8.empty());
    return out.valid;
}

bool StringParsers::parseRGBDFrame(const std::string& data, tmv::ImageFrame& out)
{
    if (!parseImageFrame(data, out)) return false;

    rapidjson::Document doc;
    if (doc.Parse(data.c_str()).HasParseError() || !doc.IsObject()) return false;

    readUint32Any(doc, "depth_stride_bytes", "depthStrideBytes", out.depth_stride_bytes);
    readFloatAny(doc, "depth_scale", "depthScale", out.depth_scale);
    if (!decodeBytesToU16Any(doc, "depth_z16", "depthZ16", out.depth_z16)) return false;

    if (out.depth_stride_bytes == 0 && out.width > 0) {
        out.depth_stride_bytes = out.width * sizeof(uint16_t);
    }
    const size_t min_depth_bytes = out.height == 0 ? 0 :
        static_cast<size_t>(out.depth_stride_bytes) * (out.height - 1) +
        static_cast<size_t>(out.width) * sizeof(uint16_t);
    const size_t actual_depth_bytes = out.depth_z16.size() * sizeof(uint16_t);
    if (out.depth_scale <= 0.0f || actual_depth_bytes < min_depth_bytes) return false;
    return true;
}

bool StringParsers::parseDepthFrame(const std::string& data, tmv::DepthFrame& out)
{
    rapidjson::Document doc;
    if (doc.Parse(data.c_str()).HasParseError() || !doc.IsObject()) {
        return false;
    }

    out = tmv::DepthFrame{};
    parseHeader(doc, out.header);
    parseIntrinsics(doc, out.intr);

    readUint32(doc, "width", out.width);
    readUint32(doc, "height", out.height);

    readUint32Any(doc, "stride_bytes", "strideBytes", out.stride_bytes);
    readFloatAny(doc, "depth_scale", "depthScale", out.depth_scale);
    readDoubleAny(doc, "rs_timestamp_ms", "rsTimestampMs", out.rs_timestamp_ms);
    parseDepthToColorCalibration(doc, out);

    if (!decodeBytesToU16Any(doc, "depth_z16", "depthZ16", out.depth_z16)) {
        return false;
    }

    if (out.stride_bytes == 0 && out.width > 0) {
        out.stride_bytes = out.width * sizeof(uint16_t);
    }

    out.valid = (out.width > 0 && out.height > 0 && !out.depth_z16.empty());
    return out.valid;
}

bool StringParsers::parseMarkerDetections(const std::string& data, tmv::MarkerDetections& out)
{
    rapidjson::Document doc;
    if (doc.Parse(data.c_str()).HasParseError() || !doc.IsObject()) {
        return false;
    }

    out = tmv::MarkerDetections{};
    parseHeader(doc, out.header);
    readUint32(doc, "width", out.width);
    readUint32(doc, "height", out.height);

    if (!doc.HasMember("markers") || !doc["markers"].IsArray()) {
        return false;
    }

    out.markers.clear();
    for (const auto& m : doc["markers"].GetArray()) {
        if (!m.IsObject()) continue;

        tmv::Marker mk;
        readInt32(m, "id", mk.id);

        if (m.HasMember("center")) parsePixelUV(m["center"], mk.center);
        if (m.HasMember("c0")) parsePixelUV(m["c0"], mk.c0);
        if (m.HasMember("c1")) parsePixelUV(m["c1"], mk.c1);
        if (m.HasMember("c2")) parsePixelUV(m["c2"], mk.c2);
        if (m.HasMember("c3")) parsePixelUV(m["c3"], mk.c3);

        if (m.HasMember("has_xyz") && m["has_xyz"].IsBool()) {
            mk.has_xyz = m["has_xyz"].GetBool();
        }

        readFloat(m, "x", mk.x);
        readFloat(m, "y", mk.y);
        readFloat(m, "z", mk.z);

        out.markers.push_back(mk);
    }

    out.valid = true;
    return true;
}

bool StringParsers::parseSafetyStatus(const std::string& data, tmv::SafetyStatus& out)
{
    rapidjson::Document doc;
    if (doc.Parse(data.c_str()).HasParseError() || !doc.IsObject()) {
        return false;
    }

    out = tmv::SafetyStatus{};
    parseHeader(doc, out.header);

    if (doc.HasMember("stop") && doc["stop"].IsBool()) out.stop = doc["stop"].GetBool();
    if (doc.HasMember("warning") && doc["warning"].IsBool()) out.warning = doc["warning"].GetBool();
    if (doc.HasMember("safe") && doc["safe"].IsBool()) out.safe = doc["safe"].GetBool();

    if (out.stop) out.level = tmv::SafetyLevel::Stop;
    else if (out.warning) out.level = tmv::SafetyLevel::Warning;
    else if (out.safe) out.level = tmv::SafetyLevel::Safe;
    else out.level = tmv::SafetyLevel::Unknown;

    out.valid = true;
    return true;
}
