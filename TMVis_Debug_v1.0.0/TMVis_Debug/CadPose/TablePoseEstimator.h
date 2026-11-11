#pragma once

#include <vector>
#include <string>
#include <memory>
#include <future>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "RgbdFrame.h"
#include "CADTemplateDetector.h"
#include "CADSceneSegmenter.h"

struct TablePose
{
    bool valid = false;
    cv::Vec3d t{0.0, 0.0, 0.0};           // CAD center position (X, Y, Z) in meters
    cv::Vec3d t_glass{0.0, 0.0, 0.0};     // Camera front-glass center (X, Y, Z)
    cv::Matx33d R = cv::Matx33d::eye();    // 3D rotation matrix
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
    float confidence = 0.0f;
    double inlier_ratio = 0.0;
    int inliers_count = 0;
    double rmse_m = 0.0;
    std::string method;
    cv::Rect2f bbox_2d;
    std::vector<cv::Point2f> corners_2d;  // 8 projected 3D corners on image
    int cluster_points_count = 0;

    // Stage-1 structural acquisition: a 2D quadrilateral for the longest
    // visible CAD frame face.  It deliberately remains separate from valid
    // 6DoF pose until enough depth exists to recover the third dimension.
    bool face_detected = false;
    float face_confidence = 0.0f;
    std::vector<cv::Point2f> face_corners_2d; // TL, TR, BR, BL
};

class TablePoseEstimator
{
public:
    TablePoseEstimator();
    ~TablePoseEstimator();

    bool initialize(const std::string& models_dir = "");

    TablePose estimate(const cv::Mat& rgb,
                       const cv::Mat& depth_z16,
                       double depth_scale,
                       const CameraIntrinsics& intr);

    // One-shot snapshot estimation: executes the exact 6-step CadViewer pipeline
    // (Gravity Floor RANSAC -> DBSCAN -> DGCNN Backbone -> Multi-Scale CAD Mesh ICP)
    TablePose estimateSnapshot(const cv::Mat& rgb,
                               const cv::Mat& depth_z16,
                               double depth_scale,
                               const CameraIntrinsics& intr);

    // Poll the asynchronous STL/hierarchy refinement and schedule the next
    // pass when its refresh period has elapsed.  This is intentionally public
    // so the snapshot UI can repaint the same captured RGB frame when the
    // cached CAD pose becomes available.
    void pollAndScheduleMeshRefinement(const cv::Mat& depth_z16,
                                       double depth_scale,
                                       const CameraIntrinsics& intr,
                                       TablePose& io_pose);

    void drawOverlay(cv::Mat& img,
                     const TablePose& pose,
                     const CameraIntrinsics& intr) const;

    // Convert a CAD->source-camera pose into another calibrated camera frame
    // (for example native depth/IR -> RGB) and regenerate its 2D projection.
    void transformPoseCameraFrame(TablePose& pose,
                                  const cv::Matx33d& target_from_source_R,
                                  const cv::Vec3d& target_from_source_t,
                                  const CameraIntrinsics& target_intr,
                                  int target_width,
                                  int target_height) const;

    bool isInitialized() const { return _initialized; }

private:
    static cv::Vec3d rotationMatrixToEulerDeg(const cv::Matx33d& R);
    static cv::Point2f project3D(const cv::Vec3d& p_cam, const CameraIntrinsics& intr);

    TablePose solveConstrainedFloorPose(const TablePose& init_pose,
                                        const cv::Vec4d& floor_plane,
                                        const std::vector<cv::Point2f>& face_corners,
                                        const std::vector<cv::Vec3f>& scene_pts,
                                        const CameraIntrinsics& intr) const;

    static bool detectLongestFrameFace(const cv::Mat& rgb,
                                       std::vector<cv::Point2f>& out_corners,
                                       float& out_confidence);

    std::vector<cv::Vec3f> extractObservationPoints(const cv::Mat& depth_z16,
                                                    double depth_scale,
                                                    const CameraIntrinsics& intr) const;

    std::vector<cv::Vec3f> extractROIPoints(const cv::Mat& depth_z16,
                                            double depth_scale,
                                            const CameraIntrinsics& intr,
                                            const cv::Rect2f& roi) const;

    std::vector<float> buildGraphFeatures(const std::vector<cv::Vec3f>& pts, int k = 20) const;

    bool runInference(const std::vector<cv::Vec3f>& obs_pts,
                      std::vector<float>& out_descriptors);

    TablePose solveRansacKabsch(const std::vector<cv::Vec3f>& obs_points,
                                const std::vector<float>& obs_descriptors,
                                float template_yaw_deg = 0.0f,
                                bool use_template_yaw = false,
                                const cv::Vec4d* floor_plane = nullptr) const;

    TablePose refinePointToPlaneGaussNewton(const TablePose& coarse_pose,
                                           const std::vector<cv::Vec3f>& obs_points,
                                           const std::vector<cv::Vec3f>& cad_interp_points,
                                           const std::vector<uint8_t>& inlier_mask) const;

    TablePose refineMultiScaleMeshICP(const TablePose& coarse_pose,
                                     const std::vector<cv::Vec3f>& scene_points,
                                     const CameraIntrinsics& intr) const;

    cv::Rect2f projectCadBoundingBox(const cv::Matx33d& R,
                                     const cv::Vec3d& t,
                                     const CameraIntrinsics& intr,
                                     int img_w, int img_h,
                                     std::vector<cv::Point2f>* out_corners_2d) const;

    // Exact refinement path used by the Qt CAD viewer: hierarchy model +
    // mesh-aware robust ICP.  It runs asynchronously and supplies a cached
    // pose, so the RGB display never waits on the ~0.8 s Python/ICP job.
    TablePose runMeshAwareRefinement(const cv::Mat& depth_z16,
                                     double depth_scale,
                                     const CameraIntrinsics& intr) const;

    struct CadFeatureEdge
    {
        cv::Vec3f p0;       // Dinh dau trong he toa do CAD (m)
        cv::Vec3f p1;       // Dinh cuoi trong he toa do CAD (m)
        cv::Vec3f n0;       // Phap tuyen mat tam giac 1
        cv::Vec3f n1;       // Phap tuyen mat tam giac 2
        float length_m = 0.0f;
    };

    // Canonical STL topology after unit conversion and alignment to the
    // normalized CAD frame used by centroids/mesh/checkpoint inference.
    struct CadTriangle
    {
        cv::Vec3f v0;
        cv::Vec3f v1;
        cv::Vec3f v2;
        cv::Vec3f normal;
    };

    bool extractStlFeatureEdges(const std::string& stl_path,
                                double feature_angle_deg = 14.0,
                                double min_edge_len_m = 0.008);
    bool unpackBlackBoxOnnx();

    bool loadUniversalCadProfile(const std::string& profile_json_path);
    void refinePoseUniversalCrease2D3D(TablePose& pose,
                                       const cv::Mat& rgb,
                                       const CameraIntrinsics& intr) const;

    TablePose refinePoseWith2DEdges(const TablePose& init_pose,
                                    const cv::Mat& rgb,
                                    const CameraIntrinsics& intr,
                                    const cv::Vec4d* floor_plane = nullptr) const;

    static bool parseNpyCentroids(const std::string& path, std::vector<cv::Vec3f>& centroids);
    static std::vector<uint8_t> readBinaryFile(const std::string& path);

private:
    std::vector<CadTriangle> _cad_triangles;
    std::vector<CadFeatureEdge> _cad_feature_edges;
    std::vector<cv::Vec3f> _cad_top_polygon;
    std::vector<cv::Vec3f> _cad_foot_centers;
    CADTemplateDetector _template_detector;
    CADSceneSegmenter _scene_segmenter;
    bool _initialized = false;

    // ONNX Runtime Neural Backbone (CPU)
    Ort::Env _ort_env;
    std::unique_ptr<Ort::Session> _ort_session;
    Ort::MemoryInfo _ort_mem_info;
    bool _ort_loaded = false;

    // Preallocated buffers for multi-input zero-copy
    std::vector<float> _f2_buf;
    std::vector<float> _f3_buf;

    // Precomputed CAD Model Data (TABLE-DEMO)
    std::vector<float> _cad_descriptors; // 512 * 128
    std::vector<cv::Vec3f> _cad_centroids; // 512
    int _num_cad_samples = 512;
    int _desc_dim = 128;
    int _num_obs_points = 1024;
    int _obs_k = 20;

    // RANSAC & Consensus Thresholds
    double _min_inlier_ratio = 0.18;
    int _min_inliers_count = 100;
    double _ransac_dist_threshold_m = 0.035;  // 35 mm consensus window
    int _ransac_max_iterations = 600;
    double _max_rmse_m = 0.050;

    // Dynamic CAD model bounding box (computed from mesh / descriptors)
    cv::Vec3d _cad_min{-0.220, -0.415, -0.3385};
    cv::Vec3d _cad_max{ 0.220,  0.415,  0.1665};

    // Table CAD model geometry
    double _half_w = 0.220;
    double _half_l = 0.415;
    double _top_z = 0.1665;
    double _beam_z = 0.0705;
    double _bot_z = -0.3385;
    double _cam_offset_x = 0.0175; // +17.5 mm Left IR to front glass centerline

    // Pre-sampled CAD mesh surface for ICP refinement (10k points with normals)
    std::vector<cv::Vec3f> _mesh_pts;     // CAD object-space positions
    std::vector<cv::Vec3f> _mesh_normals; // Outward-facing surface normals
    bool _mesh_loaded = false;

    // Self-contained Black-Box ONNX metadata
    std::string _model_name = "TABLE-DEMO";
    std::string _loaded_onnx_path;
    bool _is_table_frame = true;
    bool _is_floor_standing = true;
    bool _enable_2d_frame_anchor = true;
    std::string _topology_type = "FRAME_HOLLOW";
    double _real_bounding_diag_m = 1.07;
    std::vector<cv::Vec3f> _crease_pts_3d;
    std::vector<cv::Vec3f> _crease_normals_3d;
    int _symmetry_code = 1;
    double _plate_h = 0.0055;
    double _beam_h = 0.0455;
    double _leg_w = 0.040;
    double _foot_h = 0.046;

    std::future<TablePose> _mesh_refine_future;
    TablePose _last_mesh_pose;
    std::chrono::steady_clock::time_point _last_mesh_refine_start{};
    std::chrono::milliseconds _mesh_refine_period{1000};
};
