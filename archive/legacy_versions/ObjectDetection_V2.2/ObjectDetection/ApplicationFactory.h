#pragma once

#include <memory>
#include <string>

#include "IApplication.h"

class ApplicationFactory
{
public:
    static std::unique_ptr<IApplication> create(const std::string& application_id,
                                                std::string& error);
};
