#include "RKNNBackend.h"

#include <cstring>
#include <fstream>
#include <sstream>

RKNNBackend::~RKNNBackend()
{
    release();
}

bool RKNNBackend::loadModel(const std::string& model_path,
                            std::string& error)
{
    release();
    error.clear();

    std::vector<uint8_t> model_data = readFile(model_path);
    if (model_data.empty())
    {
        error = "cannot read RKNN model: " + model_path;
        return false;
    }

    int ret = rknn_init(&_context,
                        model_data.data(),
                        static_cast<uint32_t>(model_data.size()),
                        0,
                        nullptr);
    if (ret != RKNN_SUCC)
    {
        std::ostringstream ss;
        ss << "rknn_init failed, ret=" << ret;
        error = ss.str();
        _context = 0;
        return false;
    }

    std::memset(&_io_num, 0, sizeof(_io_num));
    ret = rknn_query(_context,
                     RKNN_QUERY_IN_OUT_NUM,
                     &_io_num,
                     sizeof(_io_num));
    if (ret != RKNN_SUCC)
    {
        error = "RKNN_QUERY_IN_OUT_NUM failed";
        release();
        return false;
    }

    if (_io_num.n_input < 1 || _io_num.n_output < 1)
    {
        std::ostringstream ss;
        ss << "unsupported RKNN IO count, input=" << _io_num.n_input
           << ", output=" << _io_num.n_output;
        error = ss.str();
        release();
        return false;
    }

    _input_attrs.resize(_io_num.n_input);
    for (uint32_t index = 0; index < _io_num.n_input; ++index)
    {
        std::memset(&_input_attrs[index], 0, sizeof(rknn_tensor_attr));
        _input_attrs[index].index = index;
        ret = rknn_query(_context,
                         RKNN_QUERY_INPUT_ATTR,
                         &_input_attrs[index],
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::ostringstream ss;
            ss << "RKNN_QUERY_INPUT_ATTR failed at index=" << index;
            error = ss.str();
            release();
            return false;
        }
    }

    _output_attrs.resize(_io_num.n_output);
    for (uint32_t index = 0; index < _io_num.n_output; ++index)
    {
        std::memset(&_output_attrs[index], 0, sizeof(rknn_tensor_attr));
        _output_attrs[index].index = index;
        ret = rknn_query(_context,
                         RKNN_QUERY_OUTPUT_ATTR,
                         &_output_attrs[index],
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::ostringstream ss;
            ss << "RKNN_QUERY_OUTPUT_ATTR failed at index=" << index;
            error = ss.str();
            release();
            return false;
        }

    }

    _model_info = BackendModelInfo{};

    for (const rknn_tensor_attr& attr : _input_attrs)
    {
        TensorInfo input_info;
        input_info.name = attr.name;
        input_info.type = convertType(attr.type);
        input_info.layout = convertLayout(attr.fmt);
        input_info.shape = copyShape(attr);
        _model_info.inputs.push_back(input_info);
    }

    for (const rknn_tensor_attr& attr : _output_attrs)
    {
        TensorInfo output_info;
        output_info.name = attr.name;
        output_info.type = TensorDataType::Float32;
        output_info.layout = convertLayout(attr.fmt);
        output_info.shape = copyShape(attr);
        _model_info.outputs.push_back(output_info);
    }

    _aux_buffers.clear();
    if (_io_num.n_input > 1)
    {
        _aux_buffers.resize(_io_num.n_input - 1);
        for (uint32_t i = 1; i < _io_num.n_input; ++i)
        {
            size_t req_size = _input_attrs[i].size > 0
                                ? static_cast<size_t>(_input_attrs[i].size)
                                : (_input_attrs[i].n_elems > 0
                                     ? static_cast<size_t>(_input_attrs[i].n_elems * sizeof(float))
                                     : tensorElementCount(copyShape(_input_attrs[i])) * sizeof(float));
            _aux_buffers[i - 1].assign(req_size, 0);
        }
    }

    _loaded = true;
    return true;
}

bool RKNNBackend::infer(const BackendInput& input,
                        std::vector<BackendOutput>& outputs,
                        std::string& error)
{
    outputs.clear();
    error.clear();

    if (!_loaded || !_context)
    {
        error = "RKNN model is not loaded";
        return false;
    }
    if (input.buffer.empty())
    {
        error = "RKNN input buffer is empty";
        return false;
    }

    rknn_tensor_type input_type;
    rknn_tensor_format input_format;
    if (!mapInputType(input.type, input_type))
    {
        error = "unsupported RKNN input data type";
        return false;
    }
    if (!mapInputLayout(input.layout, input_format))
    {
        error = "unsupported RKNN input layout";
        return false;
    }

    std::vector<rknn_input> rknn_inputs(_io_num.n_input);
    std::memset(rknn_inputs.data(), 0, sizeof(rknn_input) * _io_num.n_input);

    // Primary input (index 0): RGB image for 2D models or graph features (f1) for CAD models
    rknn_inputs[0].index = 0;
    rknn_inputs[0].type = input_type;
    rknn_inputs[0].fmt = input_format;
    rknn_inputs[0].size = static_cast<uint32_t>(input.buffer.size());
    rknn_inputs[0].buf = const_cast<uint8_t*>(input.buffer.data());
    rknn_inputs[0].pass_through = 0;

    // Auxiliary inputs for multi-input models (e.g. cad_pose f2, f3 graph features)
    if (_io_num.n_input > 1 && _aux_buffers.size() >= (_io_num.n_input - 1))
    {
        for (uint32_t i = 1; i < _io_num.n_input; ++i)
        {
            rknn_inputs[i].index = i;
            rknn_inputs[i].type = _input_attrs[i].type;
            rknn_inputs[i].fmt = _input_attrs[i].fmt;
            rknn_inputs[i].size = static_cast<uint32_t>(_aux_buffers[i - 1].size());
            rknn_inputs[i].buf = _aux_buffers[i - 1].data();
            rknn_inputs[i].pass_through = 0;
        }
    }

    int ret = rknn_inputs_set(_context, _io_num.n_input, rknn_inputs.data());
    if (ret != RKNN_SUCC)
    {
        std::ostringstream ss;
        ss << "rknn_inputs_set failed, ret=" << ret;
        error = ss.str();
        return false;
    }

    ret = rknn_run(_context, nullptr);
    if (ret != RKNN_SUCC)
    {
        std::ostringstream ss;
        ss << "rknn_run failed, ret=" << ret;
        error = ss.str();
        return false;
    }

    std::vector<rknn_output> rknn_outputs(_io_num.n_output);
    for (uint32_t index = 0; index < _io_num.n_output; ++index)
    {
        std::memset(&rknn_outputs[index], 0, sizeof(rknn_output));
        rknn_outputs[index].index = index;
        rknn_outputs[index].want_float = 1;
        rknn_outputs[index].is_prealloc = 0;
    }

    ret = rknn_outputs_get(_context,
                           _io_num.n_output,
                           rknn_outputs.data(),
                           nullptr);
    if (ret != RKNN_SUCC)
    {
        std::ostringstream ss;
        ss << "rknn_outputs_get failed, ret=" << ret;
        error = ss.str();
        return false;
    }

    outputs.reserve(_io_num.n_output);
    for (uint32_t index = 0; index < _io_num.n_output; ++index)
    {
        BackendOutput output;
        output.name = _model_info.outputs[index].name;
        output.shape = _model_info.outputs[index].shape;

        const size_t count = _output_attrs[index].n_elems > 0
                               ? static_cast<size_t>(_output_attrs[index].n_elems)
                               : tensorElementCount(output.shape);
        const float* values = static_cast<const float*>(rknn_outputs[index].buf);
        if (!values || count == 0)
        {
            rknn_outputs_release(_context,
                                 _io_num.n_output,
                                 rknn_outputs.data());
            error = "RKNN output buffer is invalid";
            return false;
        }

        output.data.assign(values, values + count);
        outputs.push_back(std::move(output));
    }

    rknn_outputs_release(_context,
                         _io_num.n_output,
                         rknn_outputs.data());
    return true;
}

void RKNNBackend::release()
{
    if (_context)
    {
        rknn_destroy(_context);
        _context = 0;
    }

    _loaded = false;
    _io_num = rknn_input_output_num{};
    _input_attrs.clear();
    _output_attrs.clear();
    _aux_buffers.clear();
    _model_info = BackendModelInfo{};
}

std::vector<uint8_t> RKNNBackend::readFile(const std::string& path)
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

TensorLayout RKNNBackend::convertLayout(rknn_tensor_format format)
{
    if (format == RKNN_TENSOR_NHWC) return TensorLayout::NHWC;
    if (format == RKNN_TENSOR_NCHW) return TensorLayout::NCHW;
    return TensorLayout::Unknown;
}

TensorDataType RKNNBackend::convertType(rknn_tensor_type type)
{
    if (type == RKNN_TENSOR_UINT8) return TensorDataType::UInt8;
    if (type == RKNN_TENSOR_FLOAT32) return TensorDataType::Float32;
    return TensorDataType::Unknown;
}

bool RKNNBackend::mapInputType(TensorDataType type, rknn_tensor_type& output)
{
    if (type == TensorDataType::UInt8)
    {
        output = RKNN_TENSOR_UINT8;
        return true;
    }
    if (type == TensorDataType::Float32)
    {
        output = RKNN_TENSOR_FLOAT32;
        return true;
    }
    return false;
}

bool RKNNBackend::mapInputLayout(TensorLayout layout, rknn_tensor_format& output)
{
    if (layout == TensorLayout::NHWC)
    {
        output = RKNN_TENSOR_NHWC;
        return true;
    }
    if (layout == TensorLayout::NCHW)
    {
        output = RKNN_TENSOR_NCHW;
        return true;
    }
    return false;
}

std::vector<int> RKNNBackend::copyShape(const rknn_tensor_attr& attr)
{
    std::vector<int> shape;
    shape.reserve(attr.n_dims);
    for (uint32_t index = 0; index < attr.n_dims; ++index)
        shape.push_back(static_cast<int>(attr.dims[index]));
    return shape;
}
