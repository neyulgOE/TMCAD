#ifndef STRING_PARSERS_H
#define STRING_PARSERS_H

#include <string>
#include "common/DataModel.h"

class StringParsers
{
public:
    static bool parseDepthFrame(const std::string& data, tmv::DepthFrame& out);
    static bool parseImageFrame(const std::string& data, tmv::ImageFrame& out);
    static bool parseRGBDFrame(const std::string& data, tmv::ImageFrame& out);
    static bool parseMarkerDetections(const std::string& data, tmv::MarkerDetections& out);
    static bool parseSafetyStatus(const std::string& data, tmv::SafetyStatus& out);
};

#endif