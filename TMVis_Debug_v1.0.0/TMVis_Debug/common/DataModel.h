#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <cstdint>
#include <array>
#include <vector>
#include <string>

namespace tmv
{
    struct Header
    {
        uint64_t pub_nsec = 0;
        uint64_t data_nsec = 0;
    };

    struct Intrinsics
    {
        uint32_t width = 0;
        uint32_t height = 0;
        float fx = 0.0f;
        float fy = 0.0f;
        float ppx = 0.0f;
        float ppy = 0.0f;
        int32_t model = 0;
        std::vector<float> coeffs;
    };

    struct PixelUV
    {
        uint32_t u = 0;
        uint32_t v = 0;
    };

    struct ImageFrame
    {
        Header header;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t stride_bytes = 0;
        double rs_timestamp_ms = 0.0;
        Intrinsics intr;
        std::vector<uint8_t> bgr8;
        // Present only on message type 11002 (CameraRGBD).  Keeping colour
        // and Z16 together preserves the V2.5 RGBD input contract.
        uint32_t depth_stride_bytes = 0;
        float depth_scale = 0.001f;
        std::vector<uint16_t> depth_z16;
        bool valid = false;

        bool hasDepth() const
        {
            return !depth_z16.empty() && depth_scale > 0.0f;
        }
    };

    struct DepthFrame
    {
        Header header;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t stride_bytes = 0;
        float depth_scale = 0.001f;
        double rs_timestamp_ms = 0.0;
        Intrinsics intr;
        // A depth image may either already be registered onto the colour
        // pixel grid, or live in the native depth/IR optical frame.  Treating
        // those two cases as identical is what produced a plausible 3D pose
        // with an incorrect RGB overlay.
        bool registered_to_color = false;
        bool has_depth_to_color_extrinsics = false;
        std::array<float, 9> depth_to_color_rotation{
            1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        std::array<float, 3> depth_to_color_translation{0.f, 0.f, 0.f};
        std::vector<uint16_t> depth_z16;
        bool valid = false;
    };

    struct Marker
    {
        int32_t id = -1;
        PixelUV center;
        PixelUV c0;
        PixelUV c1;
        PixelUV c2;
        PixelUV c3;

        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        bool has_xyz = false;
    };

    struct MarkerDetections
    {
        Header header;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<Marker> markers;
        bool valid = false;
    };

    enum class SafetyLevel
    {
        Unknown = 0,
        Safe,
        Warning,
        Stop
    };

    struct SafetyStatus
    {
        Header header;
        bool stop = false;
        bool warning = false;
        bool safe = false;
        SafetyLevel level = SafetyLevel::Unknown;
        bool valid = false;
    };
}

#endif
