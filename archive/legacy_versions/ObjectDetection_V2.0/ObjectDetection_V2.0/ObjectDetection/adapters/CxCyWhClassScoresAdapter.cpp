#include "CxCyWhClassScoresAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>

CxCyWhClassScoresAdapter::CxCyWhClassScoresAdapter(
    std::string adapter_name)
    : _adapter_name(std::move(adapter_name))
{
}

bool CxCyWhClassScoresAdapter::initialize(
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
    _num_classes = manifest.output.num_classes;

    if (_num_classes <= 0)
        _num_classes = static_cast<int>(_labels.size());

    if (_backend_info.inputs.size() != 1)
    {
        error = "cxcywh class-scores adapter requires exactly one input";
        return false;
    }
    if (_backend_info.outputs.empty())
    {
        error = "cxcywh class-scores adapter requires at least one output";
        return false;
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

    if (backend_input.shape.size() == expected_shape.size())
    {
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
    }

    if (backend_input.type != TensorDataType::Unknown &&
        backend_input.type != _manifest.input.type)
    {
        error = "manifest input type does not match backend input type";
        return false;
    }

    if (_num_classes <= 0)
    {
        error = "cxcywh class-scores class count is invalid";
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

    const TensorInfo& backend_output = _backend_info.outputs.front();
    if (!backend_output.shape.empty())
    {
        if (backend_output.shape.size() != 3 || backend_output.shape[0] != 1)
        {
            error = "cxcywh class-scores backend output must be rank 3 with batch=1";
            return false;
        }

        const int channels = 4 + _num_classes;
        const int dim1 = backend_output.shape[1];
        const int dim2 = backend_output.shape[2];
        if (dim1 != channels && dim2 != channels)
        {
            std::ostringstream ss;
            ss << "backend output shape does not match class count: [1,"
               << dim1 << ',' << dim2 << "], expected channels=" << channels;
            error = ss.str();
            return false;
        }
    }

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

    error.clear();
    return true;
}

bool CxCyWhClassScoresAdapter::preprocess(
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

bool CxCyWhClassScoresAdapter::postprocess(
    const std::vector<BackendOutput>& outputs,
    const PreprocessContext& context,
    DetectionResult& result,
    std::string& error)
{
    result = DetectionResult{};
    result.image_width = context.letterbox.src_w;
    result.image_height = context.letterbox.src_h;

    if (outputs.empty())
    {
        error = "backend returned no output";
        return false;
    }

    std::vector<RawDetection> proposals = decode(outputs.front(), context.letterbox, error);
    if (!error.empty()) return false;

    std::vector<RawDetection> kept = nms(proposals);
    if (static_cast<int>(kept.size()) > _options.max_detections)
        kept.resize(static_cast<size_t>(_options.max_detections));

    result.objects.reserve(kept.size());
    for (const RawDetection& raw : kept)
        result.objects.push_back(toDetectionObject(raw));

    return true;
}

cv::Mat CxCyWhClassScoresAdapter::letterbox(const cv::Mat& rgb, LetterboxInfo& info) const
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

bool CxCyWhClassScoresAdapter::buildInputBuffer(const cv::Mat& input_rgb,
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
        error = "cxcywh class-scores input requires 3 channels";
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
                    const float channel_mean = mean.size() > static_cast<size_t>(ch) ? mean[ch] : 0.0f;
                    const float channel_std = std.size() > static_cast<size_t>(ch) ? std[ch] : 1.0f;
                    if (std::fabs(channel_std) < 1e-12f)
                    {
                        error = "preprocess std must not be zero";
                        return false;
                    }

                    const float value = (static_cast<float>(row[x][ch]) - channel_mean) /
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
        return true;
    }

    error = "unsupported input data type";
    return false;
}

std::vector<RawDetection> CxCyWhClassScoresAdapter::decode(
    const BackendOutput& output,
    const LetterboxInfo& lb,
    std::string& error) const
{
    std::vector<RawDetection> proposals;
    error.clear();

    if (output.shape.size() != 3 || output.shape[0] != 1)
    {
        error = "cxcywh class-scores output must have shape [1,C,N] or [1,N,C]";
        return proposals;
    }

    const int dim1 = output.shape[1];
    const int dim2 = output.shape[2];
    const int channels = 4 + _num_classes;

    bool channels_first = false;
    int num_boxes = 0;
    if (dim1 == channels && dim2 > 0)
    {
        channels_first = true;
        num_boxes = dim2;
    }
    else if (dim2 == channels && dim1 > 0)
    {
        channels_first = false;
        num_boxes = dim1;
    }
    else
    {
        std::ostringstream ss;
        ss << "cxcywh class-scores output shape does not match class count: [1,"
           << dim1 << ',' << dim2 << "], expected channels=" << channels;
        error = ss.str();
        return proposals;
    }

    const size_t expected = static_cast<size_t>(channels) * num_boxes;
    if (output.data.size() < expected)
    {
        error = "cxcywh class-scores output buffer is smaller than its shape";
        return proposals;
    }

    const auto valueAt = [&](int channel, int box) -> float {
        if (channels_first)
            return output.data[static_cast<size_t>(channel) * num_boxes + box];
        return output.data[static_cast<size_t>(box) * channels + channel];
    };

    proposals.reserve(static_cast<size_t>(num_boxes) / 8U);
    for (int box = 0; box < num_boxes; ++box)
    {
        int best_class = -1;
        float best_score = 0.0f;

        for (int class_id = 0; class_id < _num_classes; ++class_id)
        {
            const float score = valueAt(4 + class_id, box);
            if (score > best_score)
            {
                best_score = score;
                best_class = class_id;
            }
        }

        if (best_class < 0 || best_score < _options.conf_threshold ||
            !isClassEnabled(best_class))
        {
            continue;
        }

        const float cx = valueAt(0, box);
        const float cy = valueAt(1, box);
        const float width = valueAt(2, box);
        const float height = valueAt(3, box);

        float x1 = (cx - width * 0.5f - lb.pad_x) / lb.scale;
        float y1 = (cy - height * 0.5f - lb.pad_y) / lb.scale;
        float x2 = (cx + width * 0.5f - lb.pad_x) / lb.scale;
        float y2 = (cy + height * 0.5f - lb.pad_y) / lb.scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(lb.src_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(lb.src_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(lb.src_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(lb.src_h - 1)));

        const float box_w = x2 - x1;
        const float box_h = y2 - y1;
        if (box_w <= 1.0f || box_h <= 1.0f) continue;

        RawDetection detection;
        detection.class_id = best_class;
        detection.score = best_score;
        detection.box = cv::Rect2f(x1, y1, box_w, box_h);
        proposals.push_back(detection);
    }

    return proposals;
}

std::vector<RawDetection> CxCyWhClassScoresAdapter::nms(
    const std::vector<RawDetection>& detections) const
{
    std::vector<RawDetection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const RawDetection& a, const RawDetection& b) {
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

float CxCyWhClassScoresAdapter::iou(const cv::Rect2f& a, const cv::Rect2f& b) const
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

bool CxCyWhClassScoresAdapter::isClassEnabled(int class_id) const
{
    if (_options.enabled_class_ids.empty()) return true;
    return std::find(_options.enabled_class_ids.begin(),
                     _options.enabled_class_ids.end(),
                     class_id) != _options.enabled_class_ids.end();
}

DetectionObject CxCyWhClassScoresAdapter::toDetectionObject(const RawDetection& raw) const
{
    DetectionObject object;
    object.class_id = raw.class_id;
    object.confidence = raw.score;
    object.label = raw.class_id >= 0 && raw.class_id < static_cast<int>(_labels.size())
                     ? _labels[raw.class_id]
                     : std::to_string(raw.class_id);

    object.x = raw.box.x;
    object.y = raw.box.y;
    object.width = raw.box.width;
    object.height = raw.box.height;
    object.center_x = object.x + object.width * 0.5f;
    object.center_y = object.y + object.height * 0.5f;
    return object;
}
