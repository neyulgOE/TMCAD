#include "ApplicationFactory.h"

#include <algorithm>
#include <cctype>

#include "applications/pallet/PalletApplication.h"
#include "applications/cad_pose/CADPoseApplication.h"

namespace
{
std::string normalized(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}
}

std::unique_ptr<IApplication> ApplicationFactory::create(
    const std::string& application_id,
    std::string& error)
{
    const std::string id = normalized(application_id);

    if (id.empty() || id == "none")
    {
        error.clear();
        return nullptr;
    }

    if (id == "pallet")
    {
        error.clear();
        return std::unique_ptr<IApplication>(new PalletApplication());
    }

    if (id == "cad_pose" || id == "cad2pose")
    {
        error.clear();
        return std::unique_ptr<IApplication>(new CADPoseApplication());
    }

    error = "unsupported ObjectDetection application: " + application_id;
    return nullptr;
}

