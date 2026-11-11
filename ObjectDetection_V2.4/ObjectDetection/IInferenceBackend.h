#pragma once

#include <string>
#include <vector>

#include "InferenceTypes.h"

class IInferenceBackend
{
public:
    virtual ~IInferenceBackend() = default;

    virtual bool loadModel(const std::string& model_path,
                           std::string& error) = 0;

    virtual bool infer(const BackendInput& input,
                       std::vector<BackendOutput>& outputs,
                       std::string& error) = 0;

    virtual void release() = 0;
    virtual bool isLoaded() const = 0;

    virtual std::string name() const = 0;
    virtual const BackendModelInfo& modelInfo() const = 0;
};
