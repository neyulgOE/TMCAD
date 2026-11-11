#include "ModelAdapterFactory.h"

#include <algorithm>
#include <cctype>

#include "adapters/CxCyWhClassScoresAdapter.h"
#include "adapters/DflMultiScaleClassScoresAdapter.h"

std::unique_ptr<IModelAdapter> ModelAdapterFactory::create(
    const std::string& adapter,
    std::string& error)
{
    std::string id = adapter;
    std::transform(id.begin(), id.end(), id.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (id == "dfl_multiscale_class_scores_v1")
    {
        error.clear();
        return std::unique_ptr<IModelAdapter>(
            new DflMultiScaleClassScoresAdapter(id));
    }

    if (id == "cxcywh_class_scores_v1")
    {
        error.clear();
        return std::unique_ptr<IModelAdapter>(
            new CxCyWhClassScoresAdapter(id));
    }

    error = "unsupported model adapter: " + adapter;
    return nullptr;
}
