#pragma once

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <future>
#include <chrono>
#include "RgbdFrame.h"
#include "CADSceneSegmenter.h"
#include "CADTemplateDetector.h"

namespace tmcad {

struct FeatureEdge3D {
    cv::Vec3f p0{0.0f, 0.0f, 0.0f};
    cv::Vec3f p1{0.0f, 0.0f, 0.0f};
    cv::Vec3f n0{0.0f, 0.0f, 1.0f};
    cv::Vec3f n1{0.0f, 0.0f, 1.0f};
    float length_m{0.0f};
};

struct ObjectPose {
    bool valid{false};
    bool face_detected{false};
    float face_confidence{0.0f};
    float confidence{0.0f};
    int inliers_count{0};
    double inlier_ratio{0.0};
    int cluster_points_count{0};
    double rmse_m{0.0};

    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0.0, 0.0, 0.0};
    cv::Vec3d t_glass{0.0, 0.0, 0.0};

    double roll_deg{0.0};
    double pitch_deg{0.0};
    double yaw_deg{0.0};

    std::string model_name{"GENERIC_CAD"};
    std::string method{"none"};
    cv::Rect2f bbox_2d;
    std::vector<cv::Point2f> corners_2d;
    std::vector<cv::Point2f> face_corners_2d;
};

class UniversalPoseEstimator {
public:
    UniversalPoseEstimator();
    ~UniversalPoseEstimator();

    // Initialize from ANY self-contained Black-Box ONNX file (<CAD_NAME>.onnx) or model directory
    bool initialize(const std::string& onnx_or_model_dir = "");

    bool isInitialized() const noexcept { return _initialized; }

    // Hot-swap or reload a different object's .onnx model at runtime without restarting the app
    bool loadModelOnnx(const std::string& onnx_path);

    // Load scale-adaptive metadata from <CAD_NAME>_profile.json (optional if embedded in Black-Box ONNX)
    bool loadUniversalProfile(const std::string& profile_json_path);
    bool loadUniversalCadProfile(const std::string& profile_json_path) {
        return loadUniversalProfile(profile_json_path);
    }

    // Single-object 6DoF estimation (supports both FLOOR_STANDING frames and FREE_6DOF parts)
    ObjectPose estimate(const cv::Mat& rgb,
                        const cv::Mat& depth_z16,
                        const CameraIntrinsics& intr);

    // Backward-compatible overload with depth_scale
    ObjectPose estimate(const cv::Mat& rgb,
                        const cv::Mat& depth_z16,
                        double depth_scale,
                        const CameraIntrinsics& intr)
    {
        (void)depth_scale;
        return estimate(rgb, depth_z16, intr);
    }

    // One-shot snapshot estimation for TMVis
    ObjectPose estimateSnapshot(const cv::Mat& rgb,
                                const cv::Mat& depth_z16,
                                double depth_scale,
                                const CameraIntrinsics& intr)
    {
        (void)depth_scale;
        return estimate(rgb, depth_z16, intr);
    }

    // TMVis async refinement polling stub
    void pollAndScheduleMeshRefinement(const cv::Mat& depth_z16,
                                       double depth_scale,
                                       const CameraIntrinsics& intr,
                                       ObjectPose& io_pose)
    {
        (void)depth_z16;
        (void)depth_scale;
        (void)intr;
        (void)io_pose;
    }

    // Multi-instance / Multi-object 6DoF estimation (e.g. multiple bottles, gears, or boxes in 1 scene)
    std::vector<ObjectPose> estimateMultiObjects(const cv::Mat& rgb,
                                                 const cv::Mat& depth_z16,
                                                 const CameraIntrinsics& intr);

    // Universal 3D HUD & Wireframe Overlay (renders any CAD's 3D crease/feature wireframe + OBB)
    void drawOverlay(cv::Mat& img,
                     const ObjectPose& pose,
                     const CameraIntrinsics& intr) const;

    void drawMultiOverlay(cv::Mat& img,
                          const std::vector<ObjectPose>& poses,
                          const CameraIntrinsics& intr) const;

    void transformPoseCameraFrame(ObjectPose& pose,
                                  const cv::Matx33d& target_from_source_R,
                                  const cv::Vec3d& target_from_source_t,
                                  const CameraIntrinsics& target_intr,
                                  int target_width,
                                  int target_height) const;

    bool isInTargetWorkspace(const cv::Vec3d& t) const;
    bool isCandidateRotationValid(const cv::Matx33d& R_candidate) const;

    const std::string& modelName() const noexcept { return _model_name; }
    bool isFloorStanding() const noexcept { return _is_floor_standing; }

private:
    bool unpackBlackBoxOnnx();
    bool unpackBlackBoxOnnxMetadata();
    bool runInference(const std::vector<cv::Vec3f>& obs_pts, std::vector<float>& out_obs_desc) const;

    ObjectPose solveRansacKabsch(const std::vector<cv::Vec3f>& obs_pts,
                                 const std::vector<float>& obs_desc,
                                 const cv::Vec4d& floor_plane,
                                 bool has_floor) const;

    void refineMultiScaleMeshICP(const std::vector<cv::Vec3f>& scene_pts,
                                 ObjectPose& pose,
                                 const CameraIntrinsics& intr) const;

    void refinePoseUniversalCrease2D3D(ObjectPose& pose,
                                       const cv::Mat& rgb,
                                       const CameraIntrinsics& intr) const;

    cv::Rect2f projectCadBoundingBox(const cv::Matx33d& R,
                                     const cv::Vec3d& t,
                                     const CameraIntrinsics& intr,
                                     int img_w,
                                     int img_h,
                                     std::vector<cv::Point2f>* out_corners = nullptr) const;

    static cv::Vec3d rotationMatrixToEulerDeg(const cv::Matx33d& R);

private:
    Ort::Env _ort_env;
    Ort::MemoryInfo _ort_mem_info;
    std::unique_ptr<Ort::Session> _ort_session;
    bool _ort_loaded{false};
    bool _initialized{false};

    std::string _model_name{"GENERIC_CAD"};
    std::string _topology_type{"COMPLEX_SOLID"};
    bool _is_floor_standing{false};
    bool _enable_2d_frame_anchor{false};

    size_t _num_obs_points{1024};
    size_t _num_cad_samples{0};
    size_t _desc_dim{128};

    double _real_bounding_diag_m{1.0};
    cv::Vec3d _cad_min{-0.220, -0.415, -0.3385};
    cv::Vec3d _cad_max{ 0.220,  0.415,  0.1665};
    double _half_w{0.220};
    double _half_l{0.415};
    double _bot_z{-0.3385};
    double _top_z{ 0.1665};

    std::vector<float> _cad_descriptors;
    std::vector<cv::Vec3f> _cad_centroids;
    std::vector<cv::Vec3f> _mesh_pts;
    std::vector<cv::Vec3f> _mesh_normals;
    std::vector<FeatureEdge3D> _cad_feature_edges;
    std::vector<cv::Vec3f> _cad_top_polygon;
    std::vector<cv::Vec3f> _cad_foot_centers;
    std::vector<cv::Matx33d> _symmetry_matrices;

    CADSceneSegmenter _scene_segmenter;
    CADTemplateDetector _template_detector;
};

// Backward-compatible aliases so existing MainWindow / TMVis code compiles without breaking
using TablePose = ObjectPose;
using TablePoseEstimator = UniversalPoseEstimator;

} // namespace tmcad

