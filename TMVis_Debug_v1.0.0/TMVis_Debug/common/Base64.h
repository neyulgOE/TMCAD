#ifndef BASE64_H
#define BASE64_H

#include <string>
#include <vector>
#include <cstdint>

class Base64
{
public:
    static bool decode(const std::string& input, std::vector<uint8_t>& output);
};

#endif