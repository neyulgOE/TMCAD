#pragma once

#include <string>
#include <vector>

#include "IInferenceBackend.h"

#if __has_include(<rknn_api.h>)
#include <rknn_api.h>
#elif __has_include("rknn_api.h")
#include "rknn_api.h"
#else
#include <cstdint>
typedef uint64_t rknn_context;
#define RKNN_SUCC 0
#define RKNN_QUERY_IN_OUT_NUM 0
#define RKNN_QUERY_INPUT_ATTR 1
#define RKNN_QUERY_OUTPUT_ATTR 2
#define RKNN_TENSOR_NHWC 0
#define RKNN_TENSOR_NCHW 1
#define RKNN_TENSOR_UINT8 0
#define RKNN_TENSOR_FLOAT32 1

typedef int rknn_tensor_format;
typedef int rknn_tensor_type;

struct rknn_input_output_num {
    uint32_t n_input;
    uint32_t n_output;
};

struct rknn_tensor_attr {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[16];
    char name[256];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
    uint8_t qnt_type;
    int8_t fl;
    int32_t zp;
    float scale;
    uint32_t w_stride;
    uint32_t size_with_stride;
    uint8_t pass_through;
    uint32_t h_stride;
};

struct rknn_input {
    uint32_t index;
    void* buf;
    uint32_t size;
    uint8_t pass_through;
    rknn_tensor_type type;
    rknn_tensor_format fmt;
};

struct rknn_output {
    uint8_t want_float;
    uint8_t is_prealloc;
    uint32_t index;
    void* buf;
    uint32_t size;
};

inline int rknn_init(rknn_context*, void*, uint32_t, uint32_t, void*) { return -1; }
inline int rknn_destroy(rknn_context) { return 0; }
inline int rknn_query(rknn_context, int, void*, uint32_t) { return -1; }
inline int rknn_inputs_set(rknn_context, uint32_t, rknn_input*) { return -1; }
inline int rknn_run(rknn_context, void*) { return -1; }
inline int rknn_outputs_get(rknn_context, uint32_t, rknn_output*, void*) { return -1; }
inline int rknn_outputs_release(rknn_context, uint32_t, rknn_output*) { return 0; }
#endif

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
    std::vector<rknn_tensor_attr> _input_attrs;
    std::vector<rknn_tensor_attr> _output_attrs;
    BackendModelInfo _model_info;
    std::vector<std::vector<uint8_t>> _aux_buffers;
};
