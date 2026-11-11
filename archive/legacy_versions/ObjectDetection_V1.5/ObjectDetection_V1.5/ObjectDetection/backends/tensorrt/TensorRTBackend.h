#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "IInferenceBackend.h"

class TensorRTBackend final : public IInferenceBackend
{
public:
    TensorRTBackend();
    ~TensorRTBackend() override;

    bool loadModel(const std::string& model_path,
                   std::string& error) override;

    bool infer(const BackendInput& input,
               std::vector<BackendOutput>& outputs,
               std::string& error) override;

    void release() override;
    bool isLoaded() const override { return _loaded; }

    std::string name() const override { return "tensorrt"; }
    const BackendModelInfo& modelInfo() const override { return _model_info; }

private:
    class Logger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* message) noexcept override;
    };

    static std::vector<uint8_t> readFile(const std::string& path);
    static size_t dataTypeSize(nvinfer1::DataType type);
    static size_t volume(const nvinfer1::Dims& dims);
    static std::vector<int> copyShape(const nvinfer1::Dims& dims);
    static TensorDataType convertType(nvinfer1::DataType type);
    static bool convertOutputToFloat(const std::vector<uint8_t>& source,
                                     nvinfer1::DataType type,
                                     std::vector<float>& destination,
                                     std::string& error);

    bool buildModelInfo(std::string& error);

private:
    Logger _logger;
    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    cudaStream_t _stream = nullptr;
    bool _loaded = false;
    BackendModelInfo _model_info;
};
