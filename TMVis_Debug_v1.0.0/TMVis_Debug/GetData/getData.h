#ifndef GETDATA_H
#define GETDATA_H

#include <string>
#include "Plugin.h"

class GetData : public Plugin
{
public:
    GetData(const std::string &name);
    ~GetData();
    void run() override;
};

#endif