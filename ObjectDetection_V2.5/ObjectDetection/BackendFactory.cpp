#include "BackendFactory.h"

#if defined(TMV_OD_BACKEND_RKNN)
#include "backends/rknn/RKNNBackend.h"
#elif defined(TMV_OD_BACKEND_TENSORRT)
#include "backends/tensorrt/TensorRTBackend.h"
#endif

std::unique_ptr<IInferenceBackend> BackendFactory::create(std::string& error)
{
    error.clear();
#if defined(TMV_OD_BACKEND_RKNN)
    return std::unique_ptr<IInferenceBackend>(new RKNNBackend());
#elif defined(TMV_OD_BACKEND_TENSORRT)
    return std::unique_ptr<IInferenceBackend>(new TensorRTBackend());
#else
    error = "ObjectDetection backend is not selected at build time";
    return nullptr;
#endif
}

std::string BackendFactory::compiledBackendName()
{
#if defined(TMV_OD_BACKEND_RKNN)
    return "rknn";
#elif defined(TMV_OD_BACKEND_TENSORRT)
    return "tensorrt";
#else
    return "unknown";
#endif
}
