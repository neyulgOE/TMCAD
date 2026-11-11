#include "TensorRTBackend.h"

#include <cuda_fp16.h>
#include <robos/core/rbs_core.h>

#include <cstring>
#include <fstream>
#include <sstream>

using namespace rbs::core;

namespace
{
struct DeviceBuffer
{
    void* ptr = nullptr;
    ~DeviceBuffer()
    {
        if (ptr) cudaFree(ptr);
    }
};
}

TensorRTBackend::TensorRTBackend() = default;

TensorRTBackend::~TensorRTBackend()
{
    release();
}

void TensorRTBackend::Logger::log(Severity severity,
                                  const char* message) noexcept
{
    if (severity <= Severity::kWARNING)
        LogError("[TensorRT] " << message);
}

bool TensorRTBackend::loadModel(const std::string& model_path,
                                std::string& error)
{
    release();
    error.clear();

    const std::vector<uint8_t> engine_data = readFile(model_path);
    if (engine_data.empty())
    {
        error = "cannot read TensorRT engine: " + model_path;
        return false;
    }

    _runtime = nvinfer1::createInferRuntime(_logger);
    if (!_runtime)
    {
        error = "createInferRuntime failed";
        return false;
    }

    _engine = _runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!_engine)
    {
        error = "deserializeCudaEngine failed";
        release();
        return false;
    }

    _context = _engine->createExecutionContext();
    if (!_context)
    {
        error = "createExecutionContext failed";
        release();
        return false;
    }

    const cudaError_t stream_ret = cudaStreamCreate(&_stream);
    if (stream_ret != cudaSuccess)
    {
        error = std::string("cudaStreamCreate failed: ") + cudaGetErrorString(stream_ret);
        release();
        return false;
    }

    if (!buildModelInfo(error))
    {
        release();
        return false;
    }

    if (_model_info.inputs.size() != 1 || _model_info.outputs.empty())
    {
        error = "ObjectDetection TensorRT backend requires one input and at least one output";
        release();
        return false;
    }

    _loaded = true;
    return true;
}

bool TensorRTBackend::infer(const BackendInput& input,
                            std::vector<BackendOutput>& outputs,
                            std::string& error)
{
    outputs.clear();
    error.clear();

    if (!_loaded || !_engine || !_context || !_stream)
    {
        error = "TensorRT engine is not loaded";
        return false;
    }
    if (input.buffer.empty())
    {
        error = "TensorRT input buffer is empty";
        return false;
    }

#if NV_TENSORRT_MAJOR >= 10
    const int io_count = _engine->getNbIOTensors();
    const char* input_name = nullptr;
    for (int index = 0; index < io_count; ++index)
    {
        const char* candidate = _engine->getIOTensorName(index);
        if (candidate &&
            _engine->getTensorIOMode(candidate) == nvinfer1::TensorIOMode::kINPUT)
        {
            input_name = candidate;
            break;
        }
    }
    if (!input_name)
    {
        error = "TensorRT input tensor name is invalid";
        return false;
    }

    const nvinfer1::Dims engine_input_dims = _engine->getTensorShape(input_name);
    bool dynamic_input = false;
    for (int i = 0; i < engine_input_dims.nbDims; ++i)
        dynamic_input = dynamic_input || engine_input_dims.d[i] < 0;

    if (dynamic_input)
    {
        nvinfer1::Dims input_dims{};
        input_dims.nbDims = static_cast<int>(input.shape.size());
        for (int i = 0; i < input_dims.nbDims; ++i) input_dims.d[i] = input.shape[i];
        if (!_context->setInputShape(input_name, input_dims))
        {
            error = "TensorRT setInputShape failed";
            return false;
        }
    }

    std::vector<DeviceBuffer> device_buffers(static_cast<size_t>(io_count));
    std::vector<std::vector<uint8_t>> host_outputs;
    std::vector<nvinfer1::DataType> output_types;
    std::vector<std::vector<int>> output_shapes;
    std::vector<std::string> output_names;

    for (int index = 0; index < io_count; ++index)
    {
        const char* name = _engine->getIOTensorName(index);
        if (!name)
        {
            error = "TensorRT IO tensor name is invalid";
            return false;
        }

        const nvinfer1::TensorIOMode mode = _engine->getTensorIOMode(name);
        const nvinfer1::DataType type = _engine->getTensorDataType(name);
        const nvinfer1::Dims dims = _context->getTensorShape(name);
        const size_t elements = volume(dims);
        const size_t bytes = elements * dataTypeSize(type);
        if (elements == 0 || bytes == 0)
        {
            error = std::string("invalid TensorRT tensor shape: ") + name;
            return false;
        }

        const cudaError_t alloc_ret = cudaMalloc(&device_buffers[index].ptr, bytes);
        if (alloc_ret != cudaSuccess)
        {
            error = std::string("cudaMalloc failed: ") + cudaGetErrorString(alloc_ret);
            return false;
        }

        if (!_context->setTensorAddress(name, device_buffers[index].ptr))
        {
            error = std::string("setTensorAddress failed: ") + name;
            return false;
        }

        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            if (bytes != input.buffer.size())
            {
                error = "TensorRT input byte size does not match engine input";
                return false;
            }
            const cudaError_t copy_ret = cudaMemcpyAsync(device_buffers[index].ptr,
                                                         input.buffer.data(),
                                                         bytes,
                                                         cudaMemcpyHostToDevice,
                                                         _stream);
            if (copy_ret != cudaSuccess)
            {
                error = std::string("input cudaMemcpyAsync failed: ") +
                        cudaGetErrorString(copy_ret);
                return false;
            }
        }
        else
        {
            host_outputs.emplace_back(bytes);
            output_types.push_back(type);
            output_shapes.push_back(copyShape(dims));
            output_names.emplace_back(name);
        }
    }

    if (!_context->enqueueV3(_stream))
    {
        error = "TensorRT enqueueV3 failed";
        return false;
    }

    size_t output_index = 0;
    for (int index = 0; index < io_count; ++index)
    {
        const char* name = _engine->getIOTensorName(index);
        if (_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) continue;

        const cudaError_t copy_ret = cudaMemcpyAsync(host_outputs[output_index].data(),
                                                     device_buffers[index].ptr,
                                                     host_outputs[output_index].size(),
                                                     cudaMemcpyDeviceToHost,
                                                     _stream);
        if (copy_ret != cudaSuccess)
        {
            error = std::string("output cudaMemcpyAsync failed: ") +
                    cudaGetErrorString(copy_ret);
            return false;
        }
        ++output_index;
    }

    const cudaError_t sync_ret = cudaStreamSynchronize(_stream);
    if (sync_ret != cudaSuccess)
    {
        error = std::string("cudaStreamSynchronize failed: ") +
                cudaGetErrorString(sync_ret);
        return false;
    }

    outputs.resize(host_outputs.size());
    for (size_t index = 0; index < host_outputs.size(); ++index)
    {
        outputs[index].name = output_names[index];
        outputs[index].shape = output_shapes[index];
        if (!convertOutputToFloat(host_outputs[index],
                                  output_types[index],
                                  outputs[index].data,
                                  error))
        {
            return false;
        }
    }
#else
    const int binding_count = _engine->getNbBindings();
    std::vector<void*> bindings(static_cast<size_t>(binding_count), nullptr);
    std::vector<DeviceBuffer> device_buffers(static_cast<size_t>(binding_count));
    std::vector<std::vector<uint8_t>> host_outputs;
    std::vector<nvinfer1::DataType> output_types;
    std::vector<std::vector<int>> output_shapes;
    std::vector<std::string> output_names;
    std::vector<int> output_bindings;

    int input_binding = -1;
    for (int index = 0; index < binding_count; ++index)
    {
        if (_engine->bindingIsInput(index))
        {
            input_binding = index;
            const nvinfer1::Dims engine_dims = _engine->getBindingDimensions(index);
            bool dynamic_input = false;
            for (int i = 0; i < engine_dims.nbDims; ++i)
                dynamic_input = dynamic_input || engine_dims.d[i] < 0;

            if (dynamic_input)
            {
                nvinfer1::Dims dims{};
                dims.nbDims = static_cast<int>(input.shape.size());
                for (int i = 0; i < dims.nbDims; ++i) dims.d[i] = input.shape[i];
                if (!_context->setBindingDimensions(index, dims))
                {
                    error = "TensorRT setBindingDimensions failed";
                    return false;
                }
            }
        }
    }

    if (input_binding < 0)
    {
        error = "TensorRT input binding not found";
        return false;
    }

    for (int index = 0; index < binding_count; ++index)
    {
        const nvinfer1::Dims dims = _context->getBindingDimensions(index);
        const nvinfer1::DataType type = _engine->getBindingDataType(index);
        const size_t elements = volume(dims);
        const size_t bytes = elements * dataTypeSize(type);
        if (elements == 0 || bytes == 0)
        {
            error = "invalid TensorRT binding shape";
            return false;
        }

        const cudaError_t alloc_ret = cudaMalloc(&device_buffers[index].ptr, bytes);
        if (alloc_ret != cudaSuccess)
        {
            error = std::string("cudaMalloc failed: ") + cudaGetErrorString(alloc_ret);
            return false;
        }
        bindings[index] = device_buffers[index].ptr;

        if (_engine->bindingIsInput(index))
        {
            if (bytes != input.buffer.size())
            {
                error = "TensorRT input byte size does not match engine input";
                return false;
            }
            const cudaError_t copy_ret = cudaMemcpyAsync(bindings[index],
                                                         input.buffer.data(),
                                                         bytes,
                                                         cudaMemcpyHostToDevice,
                                                         _stream);
            if (copy_ret != cudaSuccess)
            {
                error = std::string("input cudaMemcpyAsync failed: ") +
                        cudaGetErrorString(copy_ret);
                return false;
            }
        }
        else
        {
            host_outputs.emplace_back(bytes);
            output_types.push_back(type);
            output_shapes.push_back(copyShape(dims));
            output_names.emplace_back(_engine->getBindingName(index));
            output_bindings.push_back(index);
        }
    }

    if (!_context->enqueueV2(bindings.data(), _stream, nullptr))
    {
        error = "TensorRT enqueueV2 failed";
        return false;
    }

    for (size_t index = 0; index < output_bindings.size(); ++index)
    {
        const int binding = output_bindings[index];
        const cudaError_t copy_ret = cudaMemcpyAsync(host_outputs[index].data(),
                                                     bindings[binding],
                                                     host_outputs[index].size(),
                                                     cudaMemcpyDeviceToHost,
                                                     _stream);
        if (copy_ret != cudaSuccess)
        {
            error = std::string("output cudaMemcpyAsync failed: ") +
                    cudaGetErrorString(copy_ret);
            return false;
        }
    }

    const cudaError_t sync_ret = cudaStreamSynchronize(_stream);
    if (sync_ret != cudaSuccess)
    {
        error = std::string("cudaStreamSynchronize failed: ") +
                cudaGetErrorString(sync_ret);
        return false;
    }

    outputs.resize(host_outputs.size());
    for (size_t index = 0; index < host_outputs.size(); ++index)
    {
        outputs[index].name = output_names[index];
        outputs[index].shape = output_shapes[index];
        if (!convertOutputToFloat(host_outputs[index],
                                  output_types[index],
                                  outputs[index].data,
                                  error))
        {
            return false;
        }
    }
#endif

    return true;
}

void TensorRTBackend::release()
{
    if (_stream)
    {
        cudaStreamDestroy(_stream);
        _stream = nullptr;
    }

    delete _context;
    delete _engine;
    delete _runtime;
    _context = nullptr;
    _engine = nullptr;
    _runtime = nullptr;
    _loaded = false;
    _model_info = BackendModelInfo{};
}

std::vector<uint8_t> TensorRTBackend::readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) return {};
    return data;
}

size_t TensorRTBackend::dataTypeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(__half);
    case nvinfer1::DataType::kINT8: return sizeof(int8_t);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kBOOL: return sizeof(bool);
#endif
    default: return 0;
    }
}

size_t TensorRTBackend::volume(const nvinfer1::Dims& dims)
{
    size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0) return 0;
        result *= static_cast<size_t>(dims.d[i]);
    }
    return result;
}

std::vector<int> TensorRTBackend::copyShape(const nvinfer1::Dims& dims)
{
    std::vector<int> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) shape.push_back(dims.d[i]);
    return shape;
}

TensorDataType TensorRTBackend::convertType(nvinfer1::DataType type)
{
    if (type == nvinfer1::DataType::kFLOAT) return TensorDataType::Float32;
    return TensorDataType::Unknown;
}

bool TensorRTBackend::convertOutputToFloat(const std::vector<uint8_t>& source,
                                           nvinfer1::DataType type,
                                           std::vector<float>& destination,
                                           std::string& error)
{
    destination.clear();
    const size_t element_size = dataTypeSize(type);
    if (element_size == 0 || source.size() % element_size != 0)
    {
        error = "unsupported TensorRT output data type or size";
        return false;
    }

    const size_t count = source.size() / element_size;
    destination.resize(count);

    if (type == nvinfer1::DataType::kFLOAT)
    {
        std::memcpy(destination.data(), source.data(), source.size());
        return true;
    }
    if (type == nvinfer1::DataType::kHALF)
    {
        const __half* values = reinterpret_cast<const __half*>(source.data());
        for (size_t i = 0; i < count; ++i) destination[i] = __half2float(values[i]);
        return true;
    }
    if (type == nvinfer1::DataType::kINT8)
    {
        const int8_t* values = reinterpret_cast<const int8_t*>(source.data());
        for (size_t i = 0; i < count; ++i) destination[i] = static_cast<float>(values[i]);
        return true;
    }
    if (type == nvinfer1::DataType::kINT32)
    {
        const int32_t* values = reinterpret_cast<const int32_t*>(source.data());
        for (size_t i = 0; i < count; ++i) destination[i] = static_cast<float>(values[i]);
        return true;
    }

    error = "unsupported TensorRT output type";
    return false;
}

bool TensorRTBackend::buildModelInfo(std::string& error)
{
    _model_info = BackendModelInfo{};

#if NV_TENSORRT_MAJOR >= 10
    const int count = _engine->getNbIOTensors();
    for (int index = 0; index < count; ++index)
    {
        const char* name = _engine->getIOTensorName(index);
        if (!name) continue;

        TensorInfo info;
        info.name = name;
        info.type = convertType(_engine->getTensorDataType(name));
        info.layout = TensorLayout::Unknown;
        info.shape = copyShape(_engine->getTensorShape(name));

        if (_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            _model_info.inputs.push_back(info);
        else
            _model_info.outputs.push_back(info);
    }
#else
    const int count = _engine->getNbBindings();
    for (int index = 0; index < count; ++index)
    {
        TensorInfo info;
        info.name = _engine->getBindingName(index);
        info.type = convertType(_engine->getBindingDataType(index));
        info.layout = TensorLayout::Unknown;
        info.shape = copyShape(_engine->getBindingDimensions(index));

        if (_engine->bindingIsInput(index))
            _model_info.inputs.push_back(info);
        else
            _model_info.outputs.push_back(info);
    }
#endif

    if (_model_info.inputs.empty() || _model_info.outputs.empty())
    {
        error = "TensorRT engine has invalid IO tensors";
        return false;
    }
    return true;
}
