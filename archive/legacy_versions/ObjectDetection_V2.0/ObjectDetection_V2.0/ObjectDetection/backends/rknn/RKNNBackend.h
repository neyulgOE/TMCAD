#pragma once

#include <string>
#include <vector>

#include "IInferenceBackend.h"
#include "rknn_api.h"

class RKNNBackend final : public IInferenceBackend
{
public:
    RKNNBackend() = default;
    ~RKNNBackend() override;

    bool loadModel(const std::string& model_path,
                   std::string& error) override;

    bool infer(const BackendInput& input,
               std::vector<BackendOutput>& outputs,
               std::string& error) override;

    void release() override;
    bool isLoaded() const override { return _loaded; }

    std::string name() const override { return "rknn"; }
    const BackendModelInfo& modelInfo() const override { return _model_info; }

private:
    static std::vector<uint8_t> readFile(const std::string& path);
    static TensorLayout convertLayout(rknn_tensor_format format);
    static TensorDataType convertType(rknn_tensor_type type);
    static bool mapInputType(TensorDataType type, rknn_tensor_type& output);
    static bool mapInputLayout(TensorLayout layout, rknn_tensor_format& output);
    static std::vector<int> copyShape(const rknn_tensor_attr& attr);

private:
    rknn_context _context = 0;
    bool _loaded = false;

    rknn_input_output_num _io_num{};
    rknn_tensor_attr _input_attr{};
    std::vector<rknn_tensor_attr> _output_attrs;
    BackendModelInfo _model_info;
};
