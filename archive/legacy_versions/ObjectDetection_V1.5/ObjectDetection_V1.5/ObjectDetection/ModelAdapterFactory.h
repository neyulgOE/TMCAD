#pragma once

#include <memory>
#include <string>

#include "IModelAdapter.h"

class ModelAdapterFactory
{
public:
    static std::unique_ptr<IModelAdapter> create(const std::string& adapter,
                                                 std::string& error);
};
