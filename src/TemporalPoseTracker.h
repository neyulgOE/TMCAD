#pragma once

#include <opencv2/core.hpp>
#include <functional>
#include <vector>
#include "ObjectPose.h"
#include "CameraIntrinsics.h"

namespace tmcad {

enum class TrackState {
    SEARCHING,
    TRACKING,
    HOLD,
    LOST
};

struct PoseRefinerCallbacks {
    std::function<void(const std::vector<cv::Vec3f>&, ObjectPose&, const CameraIntrinsics&)> runLocalIcp;
    std::function<double(const cv::Mat&, const cv::Mat&, const cv::Vec4d&, bool, ObjectPose&, const CameraIntrinsics&)> runLocalSilhouette;
    std::function<bool(const std::vector<cv::Vec3f>&, const cv::Mat&, double, ObjectPose&, const CameraIntrinsics&)> evaluatePose;
    std::function<double(const cv::Mat&, const ObjectPose&, const CameraIntrinsics&)> computeDepthCoverage;
    std::function<void(const ObjectPose&, const CameraIntrinsics&, int, int, std::vector<cv::Vec3d>&)> getVisibleCadPoints;
    std::function<void(ObjectPose&, const CameraIntrinsics&, int, int)> finalizeGeometry;
};

class TemporalPoseTracker {
public:
    TemporalPoseTracker() = default;

    void configure(double cad_diag_m,
                   const cv::Vec3d& cad_min,
                   const cv::Vec3d& cad_max,
                   bool is_floor_standing,
                   const std::vector<cv::Matx33d>& symmetry_matrices);

    void setEnabled(bool enable) noexcept {
        _enabled = enable;
        if (!enable) reset();
    }
    bool isEnabled() const noexcept { return _enabled; }

    void reset() noexcept;

    TrackState state() const noexcept { return _state; }
    bool isTrackingActive() const noexcept {
        return _enabled && (_state == TrackState::TRACKING || _state == TrackState::HOLD) && _track_pose.valid;
    }
    const ObjectPose& currentPose() const noexcept { return _track_pose; }

    // Hybrid 2D-3D warm-start frame tracker (< 20 ms CPU)
    bool trackFrame(const cv::Mat& rgb,
                    const cv::Mat& depth_z16,
                    const CameraIntrinsics& intr,
                    const PoseRefinerCallbacks& cb,
                    ObjectPose& out_pose,
                    std::vector<cv::Vec3f>& out_local_fg);

    // Update FSM state from Global Detection result (handles 2-frame lock, HOLD <= 3f, LOST)
    void updateFromGlobal(ObjectPose& global_pose,
                          const cv::Mat& rgb,
                          const cv::Vec4d& floor_plane,
                          bool floor_detected,
                          const CameraIntrinsics& intr,
                          const PoseRefinerCallbacks& cb);

    // Symmetry-aware rotation distance in degrees
    double symmetryAwareRotationDiffDeg(const cv::Matx33d& R_prev, const cv::Matx33d& R_curr) const;

private:
    bool passesPoseJumpGuard(const ObjectPose& cand) const;

    void seedKeypoints2D(const cv::Mat& gray,
                         const ObjectPose& pose,
                         const CameraIntrinsics& intr,
                         const PoseRefinerCallbacks& cb);

    bool trackKeypointsPyrLKAndPnP(const cv::Mat& curr_gray,
                                   const CameraIntrinsics& intr,
                                   cv::Matx33d& out_R,
                                   cv::Vec3d& out_t);

    bool _enabled{false};
    TrackState _state{TrackState::SEARCHING};
    int _confirm_streak{0};
    int _lost_streak{0};
    int _weak_frames{0};
    int _tracked_frames{0};

    double _cad_diag_m{0.20};
    cv::Vec3d _cad_min{-0.1, -0.1, -0.1};
    cv::Vec3d _cad_max{ 0.1,  0.1,  0.1};
    bool _is_floor_standing{false};
    std::vector<cv::Matx33d> _symmetry_matrices;

    bool _has_floor{false};
    cv::Vec4d _floor_plane{0.0, -1.0, 0.0, 0.0};
    ObjectPose _track_pose;
    ObjectPose _prev_global_pose;

    // 2D sparse PyrLK + PnP state (lightweight Point2Pose for C++)
    cv::Mat _prev_gray;
    std::vector<cv::Point2f> _kpts_2d;
    std::vector<cv::Point3f> _kpts_3d_cad;
};

} // namespace tmcad

