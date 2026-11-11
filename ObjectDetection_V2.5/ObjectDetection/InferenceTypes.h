#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "DetectionTypes.h"

enum class TensorDataType
{
    Unknown = 0,
    UInt8,
    Float32
};

enum class TensorLayout
{
    Unknown = 0,
    NHWC,
    NCHW
};

struct TensorInfo
{
    std::string name;
    TensorDataType type = TensorDataType::Unknown;
    TensorLayout layout = TensorLayout::Unknown;
    std::vector<int> shape;
};

struct BackendModelInfo
{
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
};

struct BackendInput
{
    std::string name;
    TensorDataType type = TensorDataType::Unknown;
    TensorLayout layout = TensorLayout::Unknown;
    std::vector<int> shape;
    std::vector<uint8_t> buffer;
};

struct BackendOutput
{
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
};

struct PreprocessContext
{
    LetterboxInfo letterbox;
};

inline size_t tensorElementCount(const std::vector<int>& shape)
{
    size_t count = 1;
    for (int dim : shape)
    {
        if (dim <= 0) return 0;
        count *= static_cast<size_t>(dim);
    }
    return count;
}
