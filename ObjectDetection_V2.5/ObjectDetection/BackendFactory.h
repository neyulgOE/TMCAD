#pragma once

#include <memory>
#include <string>

#include "IInferenceBackend.h"

class BackendFactory
{
public:
    static std::unique_ptr<IInferenceBackend> create(std::string& error);
    static std::string compiledBackendName();
};
