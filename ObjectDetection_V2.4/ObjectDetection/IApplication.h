#pragma once

#include <string>

#include "DetectionTypes.h"
#include "ObjectDetectionConfig.h"
#include "RgbdFrame.h"

class IApplication
{
public:
    virtual ~IApplication() = default;

    virtual std::string name() const = 0;
    virtual bool requiresRgbd() const { return true; }

    virtual bool initialize(const DetectorConfig& config,
                            std::string& error) = 0;

    virtual bool process(const RgbdFrame& frame,
                         DetectorOutput& output,
                         std::string& error) = 0;
};
