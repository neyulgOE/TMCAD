#include "DflMultiScaleClassScoresAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace
{
std::string shapeToString(const std::vector<int>& shape)
{
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0) ss << ',';
        ss << shape[i];
    }
    ss << ']';
    return ss.str();
}

std::string intVectorToString(const std::vector<int>& values)
{
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0) ss << ',';
        ss << values[i];
    }
    ss << ']';
    return ss.str();
}

const char* layoutName(TensorLayout layout)
{
    switch (layout)
    {
        case TensorLayout::NHWC: return "NHWC";
        case TensorLayout::NCHW: return "NCHW";
        default: return "Unknown";
    }
}

}

DflMultiScaleClassScoresAdapter::DflMultiScaleClassScoresAdapter(
    std::string adapter_name)
    : _adapter_name(std::move(adapter_name))
{
}

bool DflMultiScaleClassScoresAdapter::initialize(
    const ModelManifest& manifest,
    const BackendModelInfo& backend_info,
    const std::vector<std::string>& labels,
    const AdapterRuntimeOptions& options,
    std::string& error)
{
    _manifest = manifest;
    _backend_info = backend_info;
    _labels = labels;
    _options = options;

    _num_classes = _manifest.output.num_classes;
    if (_num_classes <= 0)
        _num_classes = static_cast<int>(_labels.size());

    _reg_max = _manifest.output.reg_max;
    _strides = _manifest.output.strides;

    if (_backend_info.inputs.size() != 1)
    {
        error = "DFL multi-scale adapter requires exactly one input";
        return false;
    }
    if (_backend_info.outputs.empty())
    {
        error = "DFL multi-scale adapter requires output tensors";
        return false;
    }
    if (_num_classes <= 0)
    {
        error = "DFL multi-scale class count is invalid";
        return false;
    }
    if (static_cast<int>(_labels.size()) < _num_classes)
    {
        std::ostringstream ss;
        ss << "labels count is smaller than num_classes: labels="
           << _labels.size() << ", num_classes=" << _num_classes;
        error = ss.str();
        return false;
    }
    if (_reg_max <= 1)
    {
        error = "DFL multi-scale reg_max must be greater than 1";
        return false;
    }
    if (_strides.empty())
    {
        error = "DFL multi-scale strides are missing from model manifest";
        return false;
    }
    if (!_manifest.output.class_scores_activated)
    {
        error = "DFL multi-scale v1 requires class_scores_activated=true";
        return false;
    }

    std::set<int> unique_strides;
    for (int stride : _strides)
    {
        if (stride <= 0)
        {
            error = "DFL multi-scale stride must be positive";
            return false;
        }
        if (!unique_strides.insert(stride).second)
        {
            error = "DFL multi-scale strides must be unique";
            return false;
        }
    }

    const TensorInfo& backend_input = _backend_info.inputs.front();
    std::vector<int> expected_shape;
    if (_manifest.input.layout == TensorLayout::NHWC)
    {
        expected_shape = {1,
                          _manifest.input.height,
                          _manifest.input.width,
                          _manifest.input.channels};
    }
    else if (_manifest.input.layout == TensorLayout::NCHW)
    {
        expected_shape = {1,
                          _manifest.input.channels,
                          _manifest.input.height,
                          _manifest.input.width};
    }
    else
    {
        error = "DFL multi-scale manifest input layout is unsupported";
        return false;
    }

    if (backend_input.shape.size() != expected_shape.size())
    {
        error = "DFL multi-scale backend input rank does not match manifest";
        return false;
    }
    for (size_t index = 0; index < expected_shape.size(); ++index)
    {
        if (backend_input.shape[index] > 0 &&
            backend_input.shape[index] != expected_shape[index])
        {
            std::ostringstream ss;
            ss << "manifest input shape does not match backend at dim=" << index
               << ", manifest=" << expected_shape[index]
               << ", backend=" << backend_input.shape[index];
            error = ss.str();
            return false;
        }
    }

    // RKNN internal input type can differ from the UINT8 runtime contract.

    for (int class_id : _options.enabled_class_ids)
    {
        if (class_id < 0 || class_id >= _num_classes)
        {
            std::ostringstream ss;
            ss << "enabled class id out of range: " << class_id;
            error = ss.str();
            return false;
        }
    }

    if (!buildStrideGroups(error))
        return false;

    error.clear();
    return true;
}

bool DflMultiScaleClassScoresAdapter::preprocess(
    const cv::Mat& rgb,
    BackendInput& input,
    PreprocessContext& context,
    std::string& error)
{
    if (rgb.empty() || rgb.type() != CV_8UC3)
    {
        error = "input image must be non-empty CV_8UC3 RGB";
        return false;
    }
    if (_manifest.preprocess.type != "letterbox")
    {
        error = "unsupported preprocess type: " + _manifest.preprocess.type;
        return false;
    }

    cv::Mat input_rgb = letterbox(rgb, context.letterbox);
    return buildInputBuffer(input_rgb, input, error);
}

bool DflMultiScaleClassScoresAdapter::postprocess(
    const std::vector<BackendOutput>& outputs,
    const PreprocessContext& context,
    DetectionResult& result,
    std::string& error)
{
    result = DetectionResult{};
    result.image_width = context.letterbox.src_w;
    result.image_height = context.letterbox.src_h;

    if (outputs.size() != _backend_info.outputs.size())
    {
        std::ostringstream ss;
        ss << "backend output count changed after adapter initialization: expected="
           << _backend_info.outputs.size() << ", actual=" << outputs.size();
        error = ss.str();
        return false;
    }

    std::vector<RawDetection> proposals =
        decodeAll(outputs, context.letterbox, error);
    if (!error.empty()) return false;

    std::vector<RawDetection> kept = nms(proposals);
    if (static_cast<int>(kept.size()) > _options.max_detections)
        kept.resize(static_cast<size_t>(_options.max_detections));

    result.objects.reserve(kept.size());
    for (const RawDetection& raw : kept)
        result.objects.push_back(toDetectionObject(raw));

    error.clear();
    return true;
}

bool DflMultiScaleClassScoresAdapter::buildStrideGroups(std::string& error)
{
    struct GroupBuild
    {
        int grid_width = 0;
        int grid_height = 0;
        int box_index = -1;
        int cls_index = -1;
        int aux_index = -1;
        TensorLayout box_layout = TensorLayout::Unknown;
        TensorLayout cls_layout = TensorLayout::Unknown;
    };

    std::map<std::pair<int, int>, GroupBuild> grouped;
    const int box_channels = 4 * _reg_max;

    for (size_t index = 0; index < _backend_info.outputs.size(); ++index)
    {
        const TensorInfo& info = _backend_info.outputs[index];
        int channels = 0;
        int height = 0;
        int width = 0;
        if (!tensorCHW(info, channels, height, width))
        {
            std::ostringstream ss;
            ss << "cannot determine C/H/W for DFL output index=" << index
               << " name=" << info.name
               << " shape=" << shapeToString(info.shape)
               << " layout=" << layoutName(info.layout);
            error = ss.str();
            return false;
        }

        const std::pair<int, int> key(height, width);
        GroupBuild& group = grouped[key];
        group.grid_width = width;
        group.grid_height = height;

        if (channels == box_channels)
        {
            if (group.box_index >= 0)
            {
                error = "duplicate DFL box tensor for the same grid";
                return false;
            }
            group.box_index = static_cast<int>(index);
            group.box_layout = info.layout;
        }
        else if (channels == _num_classes)
        {
            if (group.cls_index >= 0)
            {
                error = "duplicate DFL class tensor for the same grid";
                return false;
            }
            group.cls_index = static_cast<int>(index);
            group.cls_layout = info.layout;
        }
        else if (channels == 1)
        {
            if (group.aux_index >= 0)
            {
                error = "duplicate DFL auxiliary tensor for the same grid";
                return false;
            }
            group.aux_index = static_cast<int>(index);
        }
        else
        {
            std::ostringstream ss;
            ss << "unexpected DFL output channels=" << channels
               << " at index=" << index
               << ", expected " << box_channels
               << " (box), " << _num_classes
               << " (class), or 1 (auxiliary)";
            error = ss.str();
            return false;
        }
    }

    _groups.clear();
    std::set<int> resolved_strides;

    for (const auto& item : grouped)
    {
        const GroupBuild& source = item.second;
        if (source.box_index < 0 || source.cls_index < 0)
        {
            std::ostringstream ss;
            ss << "DFL grid " << source.grid_width << 'x' << source.grid_height
               << " is missing box or class tensor";
            error = ss.str();
            return false;
        }

        if (_manifest.input.width % source.grid_width != 0 ||
            _manifest.input.height % source.grid_height != 0)
        {
            error = "DFL grid does not divide model input dimensions";
            return false;
        }

        const int stride_x = _manifest.input.width / source.grid_width;
        const int stride_y = _manifest.input.height / source.grid_height;
        if (stride_x != stride_y)
        {
            error = "DFL grid has different X/Y stride";
            return false;
        }
        if (std::find(_strides.begin(), _strides.end(), stride_x) == _strides.end())
        {
            std::ostringstream ss;
            ss << "DFL output resolved unexpected stride=" << stride_x
               << ", manifest strides=" << intVectorToString(_strides);
            error = ss.str();
            return false;
        }
        if (!resolved_strides.insert(stride_x).second)
        {
            error = "duplicate DFL stride group";
            return false;
        }

        StrideGroup group;
        group.stride = stride_x;
        group.grid_width = source.grid_width;
        group.grid_height = source.grid_height;
        group.box_output_index = source.box_index;
        group.cls_output_index = source.cls_index;
        group.aux_output_index = source.aux_index;
        group.box_layout = source.box_layout;
        group.cls_layout = source.cls_layout;
        _groups.push_back(group);
    }

    if (_groups.size() != _strides.size())
    {
        std::ostringstream ss;
        ss << "DFL stride group count mismatch: groups=" << _groups.size()
           << ", manifest=" << _strides.size();
        error = ss.str();
        return false;
    }

    for (int stride : _strides)
    {
        if (resolved_strides.find(stride) == resolved_strides.end())
        {
            std::ostringstream ss;
            ss << "DFL manifest stride not found in backend outputs: " << stride;
            error = ss.str();
            return false;
        }
    }

    std::sort(_groups.begin(), _groups.end(), [](const StrideGroup& a,
                                                  const StrideGroup& b) {
        return a.stride < b.stride;
    });

    error.clear();
    return true;
}

bool DflMultiScaleClassScoresAdapter::tensorCHW(
    const TensorInfo& info,
    int& channels,
    int& height,
    int& width) const
{
    channels = 0;
    height = 0;
    width = 0;

    if (info.shape.size() != 4 || info.shape[0] != 1)
        return false;

    if (info.layout == TensorLayout::NCHW)
    {
        channels = info.shape[1];
        height = info.shape[2];
        width = info.shape[3];
        return channels > 0 && height > 0 && width > 0;
    }
    if (info.layout == TensorLayout::NHWC)
    {
        height = info.shape[1];
        width = info.shape[2];
        channels = info.shape[3];
        return channels > 0 && height > 0 && width > 0;
    }

    // Unknown output layout is rejected instead of inferred.
    return false;
}

float DflMultiScaleClassScoresAdapter::elementAt(
    const BackendOutput& output,
    TensorLayout layout,
    int channels,
    int height,
    int width,
    int channel,
    int y,
    int x) const
{
    size_t index = 0;
    if (layout == TensorLayout::NCHW)
    {
        index = (static_cast<size_t>(channel) * height + y) * width + x;
    }
    else
    {
        index = (static_cast<size_t>(y) * width + x) * channels + channel;
    }
    return output.data[index];
}

float DflMultiScaleClassScoresAdapter::decodeDflDistance(
    const BackendOutput& box_output,
    TensorLayout layout,
    int height,
    int width,
    int side,
    int y,
    int x,
    bool& ok) const
{
    ok = false;
    const int box_channels = 4 * _reg_max;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int bin = 0; bin < _reg_max; ++bin)
    {
        const int channel = side * _reg_max + bin;
        const float value = elementAt(box_output,
                                      layout,
                                      box_channels,
                                      height,
                                      width,
                                      channel,
                                      y,
                                      x);
        if (!std::isfinite(value)) return 0.0f;
        max_logit = std::max(max_logit, value);
    }

    double sum_exp = 0.0;
    double weighted = 0.0;
    for (int bin = 0; bin < _reg_max; ++bin)
    {
        const int channel = side * _reg_max + bin;
        const float value = elementAt(box_output,
                                      layout,
                                      box_channels,
                                      height,
                                      width,
                                      channel,
                                      y,
                                      x);
        const double e = std::exp(static_cast<double>(value - max_logit));
        sum_exp += e;
        weighted += e * static_cast<double>(bin);
    }

    if (!std::isfinite(sum_exp) || sum_exp <= 0.0 ||
        !std::isfinite(weighted))
    {
        return 0.0f;
    }

    ok = true;
    return static_cast<float>(weighted / sum_exp);
}

std::vector<RawDetection> DflMultiScaleClassScoresAdapter::decodeAll(
    const std::vector<BackendOutput>& outputs,
    const LetterboxInfo& lb,
    std::string& error) const
{
    std::vector<RawDetection> proposals;
    proposals.reserve(512);

    for (const StrideGroup& group : _groups)
    {
        if (group.box_output_index < 0 || group.cls_output_index < 0 ||
            static_cast<size_t>(group.box_output_index) >= outputs.size() ||
            static_cast<size_t>(group.cls_output_index) >= outputs.size())
        {
            error = "DFL stride group output index is invalid";
            return {};
        }

        const BackendOutput& box_output =
            outputs[static_cast<size_t>(group.box_output_index)];
        const BackendOutput& cls_output =
            outputs[static_cast<size_t>(group.cls_output_index)];

        const size_t expected_box = static_cast<size_t>(4 * _reg_max) *
                                    group.grid_height * group.grid_width;
        const size_t expected_cls = static_cast<size_t>(_num_classes) *
                                    group.grid_height * group.grid_width;
        if (box_output.data.size() != expected_box)
        {
            std::ostringstream ss;
            ss << "DFL box buffer size mismatch for stride=" << group.stride
               << ": expected=" << expected_box
               << ", actual=" << box_output.data.size();
            error = ss.str();
            return {};
        }
        if (cls_output.data.size() != expected_cls)
        {
            std::ostringstream ss;
            ss << "DFL class buffer size mismatch for stride=" << group.stride
               << ": expected=" << expected_cls
               << ", actual=" << cls_output.data.size();
            error = ss.str();
            return {};
        }

        for (int gy = 0; gy < group.grid_height; ++gy)
        {
            for (int gx = 0; gx < group.grid_width; ++gx)
            {
                int best_class = -1;
                float best_score = -std::numeric_limits<float>::infinity();

                for (int class_id = 0; class_id < _num_classes; ++class_id)
                {
                    const float score = elementAt(cls_output,
                                                  group.cls_layout,
                                                  _num_classes,
                                                  group.grid_height,
                                                  group.grid_width,
                                                  class_id,
                                                  gy,
                                                  gx);
                    if (!std::isfinite(score))
                        continue;

                    if (score > best_score)
                    {
                        best_score = score;
                        best_class = class_id;
                    }
                }

                if (best_class < 0 || !std::isfinite(best_score) ||
                    best_score < _options.conf_threshold ||
                    !isClassEnabled(best_class))
                {
                    continue;
                }

                float dist[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                bool valid_dfl = true;
                for (int side = 0; side < 4; ++side)
                {
                    bool side_ok = false;
                    dist[side] = decodeDflDistance(box_output,
                                                   group.box_layout,
                                                   group.grid_height,
                                                   group.grid_width,
                                                   side,
                                                   gy,
                                                   gx,
                                                   side_ok);
                    if (!side_ok)
                    {
                        valid_dfl = false;
                        break;
                    }
                }
                if (!valid_dfl)
                    continue;

                const float cx_grid = static_cast<float>(gx) + 0.5f;
                const float cy_grid = static_cast<float>(gy) + 0.5f;

                const float input_x1 =
                    (cx_grid - dist[0]) * static_cast<float>(group.stride);
                const float input_y1 =
                    (cy_grid - dist[1]) * static_cast<float>(group.stride);
                const float input_x2 =
                    (cx_grid + dist[2]) * static_cast<float>(group.stride);
                const float input_y2 =
                    (cy_grid + dist[3]) * static_cast<float>(group.stride);

                if (!std::isfinite(input_x1) || !std::isfinite(input_y1) ||
                    !std::isfinite(input_x2) || !std::isfinite(input_y2) ||
                    input_x2 <= input_x1 || input_y2 <= input_y1)
                {
                    continue;
                }

                float x1 = (input_x1 - static_cast<float>(lb.pad_x)) / lb.scale;
                float y1 = (input_y1 - static_cast<float>(lb.pad_y)) / lb.scale;
                float x2 = (input_x2 - static_cast<float>(lb.pad_x)) / lb.scale;
                float y2 = (input_y2 - static_cast<float>(lb.pad_y)) / lb.scale;

                x1 = std::max(0.0f, std::min(x1, static_cast<float>(lb.src_w)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(lb.src_h)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(lb.src_w)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(lb.src_h)));

                const float box_w = x2 - x1;
                const float box_h = y2 - y1;
                if (box_w <= 1.0f || box_h <= 1.0f) continue;

                RawDetection detection;
                detection.class_id = best_class;
                detection.score = best_score;
                detection.box = cv::Rect2f(x1, y1, box_w, box_h);
                proposals.push_back(detection);
            }
        }

    }

    error.clear();
    return proposals;
}

cv::Mat DflMultiScaleClassScoresAdapter::letterbox(
    const cv::Mat& rgb,
    LetterboxInfo& info) const
{
    const int dst_w = _manifest.input.width;
    const int dst_h = _manifest.input.height;

    info.src_w = rgb.cols;
    info.src_h = rgb.rows;
    info.input_w = dst_w;
    info.input_h = dst_h;

    const float scale = std::min(dst_w / static_cast<float>(rgb.cols),
                                 dst_h / static_cast<float>(rgb.rows));
    const int new_w = std::max(1, static_cast<int>(std::round(rgb.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(rgb.rows * scale)));

    info.scale = scale;
    info.pad_x = (dst_w - new_w) / 2;
    info.pad_y = (dst_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h));

    const std::vector<int>& padding = _manifest.preprocess.padding_value;
    const int p0 = padding.empty() ? 114 : padding[0];
    const int p1 = padding.size() > 1 ? padding[1] : p0;
    const int p2 = padding.size() > 2 ? padding[2] : p0;

    cv::Mat output(dst_h, dst_w, CV_8UC3, cv::Scalar(p0, p1, p2));
    resized.copyTo(output(cv::Rect(info.pad_x, info.pad_y, new_w, new_h)));
    return output;
}

bool DflMultiScaleClassScoresAdapter::buildInputBuffer(
    const cv::Mat& input_rgb,
    BackendInput& input,
    std::string& error) const
{
    input = BackendInput{};
    input.name = _backend_info.inputs.front().name;
    input.type = _manifest.input.type;
    input.layout = _manifest.input.layout;

    const int h = input_rgb.rows;
    const int w = input_rgb.cols;
    const int c = input_rgb.channels();

    if (c != 3)
    {
        error = "DFL multi-scale input requires 3 channels";
        return false;
    }

    if (input.layout == TensorLayout::NHWC)
        input.shape = {1, h, w, c};
    else if (input.layout == TensorLayout::NCHW)
        input.shape = {1, c, h, w};
    else
    {
        error = "unsupported input layout";
        return false;
    }

    if (input.type == TensorDataType::UInt8)
    {
        const size_t bytes = static_cast<size_t>(h) * static_cast<size_t>(w) * 3U;
        input.buffer.resize(bytes);

        if (input.layout == TensorLayout::NHWC)
        {
            if (input_rgb.isContinuous())
            {
                std::memcpy(input.buffer.data(), input_rgb.data, bytes);
            }
            else
            {
                for (int y = 0; y < h; ++y)
                {
                    std::memcpy(input.buffer.data() + static_cast<size_t>(y) * w * 3U,
                                input_rgb.ptr(y),
                                static_cast<size_t>(w) * 3U);
                }
            }
        }
        else
        {
            for (int y = 0; y < h; ++y)
            {
                const cv::Vec3b* row = input_rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < w; ++x)
                {
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        const size_t index = static_cast<size_t>(ch) * h * w +
                                             static_cast<size_t>(y) * w + x;
                        input.buffer[index] = row[x][ch];
                    }
                }
            }
        }
        error.clear();
        return true;
    }

    if (input.type == TensorDataType::Float32)
    {
        const size_t count = static_cast<size_t>(h) * static_cast<size_t>(w) * 3U;
        input.buffer.resize(count * sizeof(float));
        float* dst = reinterpret_cast<float*>(input.buffer.data());

        const std::vector<float>& mean = _manifest.preprocess.mean;
        const std::vector<float>& std = _manifest.preprocess.std;

        for (int y = 0; y < h; ++y)
        {
            const cv::Vec3b* row = input_rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; ++x)
            {
                for (int ch = 0; ch < 3; ++ch)
                {
                    const float channel_mean =
                        mean.size() > static_cast<size_t>(ch) ? mean[ch] : 0.0f;
                    const float channel_std =
                        std.size() > static_cast<size_t>(ch) ? std[ch] : 1.0f;
                    if (std::fabs(channel_std) < 1e-12f)
                    {
                        error = "preprocess std must not be zero";
                        return false;
                    }

                    const float value =
                        (static_cast<float>(row[x][ch]) - channel_mean) /
                        channel_std;
                    size_t index = 0;
                    if (input.layout == TensorLayout::NHWC)
                    {
                        index = (static_cast<size_t>(y) * w + x) * 3U + ch;
                    }
                    else
                    {
                        index = static_cast<size_t>(ch) * h * w +
                                static_cast<size_t>(y) * w + x;
                    }
                    dst[index] = value;
                }
            }
        }
        error.clear();
        return true;
    }

    error = "unsupported input data type";
    return false;
}

std::vector<RawDetection> DflMultiScaleClassScoresAdapter::nms(
    const std::vector<RawDetection>& detections) const
{
    std::vector<RawDetection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const RawDetection& a,
                                                const RawDetection& b) {
        return a.score > b.score;
    });

    std::vector<RawDetection> kept;
    std::vector<bool> removed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i)
    {
        if (removed[i]) continue;
        kept.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); ++j)
        {
            if (removed[j]) continue;
            if (sorted[i].class_id == sorted[j].class_id &&
                iou(sorted[i].box, sorted[j].box) > _options.nms_threshold)
            {
                removed[j] = true;
            }
        }
    }

    return kept;
}

float DflMultiScaleClassScoresAdapter::iou(
    const cv::Rect2f& a,
    const cv::Rect2f& b) const
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0f, x2 - x1);
    const float inter_h = std::max(0.0f, y2 - y1);
    const float intersection = inter_w * inter_h;
    const float union_area = a.width * a.height + b.width * b.height - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

bool DflMultiScaleClassScoresAdapter::isClassEnabled(int class_id) const
{
    if (_options.enabled_class_ids.empty()) return true;
    return std::find(_options.enabled_class_ids.begin(),
                     _options.enabled_class_ids.end(),
                     class_id) != _options.enabled_class_ids.end();
}

DetectionObject DflMultiScaleClassScoresAdapter::toDetectionObject(
    const RawDetection& raw) const
{
    DetectionObject object;
    object.class_id = raw.class_id;
    object.confidence = raw.score;
    object.label = raw.class_id >= 0 &&
                           raw.class_id < static_cast<int>(_labels.size())
                       ? _labels[static_cast<size_t>(raw.class_id)]
                       : ("class_" + std::to_string(raw.class_id));

    object.x = raw.box.x;
    object.y = raw.box.y;
    object.width = raw.box.width;
    object.height = raw.box.height;
    object.center_x = raw.box.x + raw.box.width * 0.5f;
    object.center_y = raw.box.y + raw.box.height * 0.5f;
    return object;
}
