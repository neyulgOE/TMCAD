#include "UniversalPoseEstimator.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

namespace tmcad {

namespace {
constexpr double kPi = 3.14159265358979323846;

bool fileExists(const std::string& p) {
    return !p.empty() && std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
}

std::string extractStem(const std::string& p) {
    return std::filesystem::path(p).stem().string();
}
} // anonymous namespace

UniversalPoseEstimator::UniversalPoseEstimator()
    : _ort_env(ORT_LOGGING_LEVEL_WARNING, "TMCAD_UniversalPose"),
      _ort_mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    _symmetry_matrices.push_back(cv::Matx33d::eye());
}

UniversalPoseEstimator::~UniversalPoseEstimator()
{
    _ort_session.reset();
}

bool UniversalPoseEstimator::initialize(const std::string& onnx_or_model_dir)
{
    std::string req = onnx_or_model_dir;
    if (req.empty()) {
        const char* env_model = std::getenv("TMCAD_ONNX_MODEL");
        if (env_model && env_model[0] != '\0') {
            req = std::string(env_model);
        } else {
            req = "TABLE-DEMO";
        }
    }

    // 1. Dynamically discover any .onnx file in the provided path or models/ directory
    std::vector<std::string> candidates;
    if (!req.empty()) {
        if (fileExists(req) && req.size() > 5 && req.substr(req.size() - 5) == ".onnx") {
            candidates.push_back(req);
        } else {
            candidates.push_back(req + ".onnx");
            candidates.push_back("models/" + req + ".onnx");
            candidates.push_back("deploy_orange_pi_5b/" + req + "/" + req + ".onnx");
            candidates.push_back(req + "/model_obs_npu.onnx");
        }
    }

    // Scan models/ folder dynamically for any available .onnx if req was not explicitly found
    if (std::filesystem::exists("models") && std::filesystem::is_directory("models")) {
        for (const auto& entry : std::filesystem::directory_iterator("models")) {
            if (entry.path().extension() == ".onnx") {
                candidates.push_back(entry.path().string());
            }
        }
    }

    for (const auto& cand : candidates) {
        if (fileExists(cand) && loadModelOnnx(cand)) {
            return true;
        }
    }

    std::cerr << "[UniversalPoseEstimator] No valid .onnx model found for request: " << req << std::endl;
    return false;
}

bool UniversalPoseEstimator::loadModelOnnx(const std::string& onnx_path)
{
    if (!fileExists(onnx_path)) return false;

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        _ort_session = std::make_unique<Ort::Session>(_ort_env, onnx_path.c_str(), opts);
        _ort_loaded = true;
        _model_name = extractStem(onnx_path);

        if (_model_name == "model_obs_npu") {
            auto parent = std::filesystem::path(onnx_path).parent_path().filename().string();
            if (!parent.empty() && parent != "models") _model_name = parent;
        }

        if (unpackBlackBoxOnnx()) {
            unpackBlackBoxOnnxMetadata();
            _initialized = true;
            std::cout << "[UniversalPoseEstimator] Ready with Black-Box ONNX (" << _model_name
                      << ") | Mode: " << (_is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF")
                      << " | Diag: " << _real_bounding_diag_m << " m" << std::endl;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[UniversalPoseEstimator] ONNX load error: " << e.what() << std::endl;
        _ort_loaded = false;
    }
    return false;
}

bool UniversalPoseEstimator::unpackBlackBoxOnnx()
{
    if (!_ort_loaded || !_ort_session) return false;
    size_t num_outputs = _ort_session->GetOutputCount();
    if (num_outputs < 6) return false;

    std::vector<float> dummy_in(_num_obs_points * 3, 0.0f);
    std::array<int64_t, 3> in_shape{1, static_cast<int64_t>(_num_obs_points), 3};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        _ort_mem_info, dummy_in.data(), dummy_in.size(), in_shape.data(), in_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = _ort_session->GetInputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};

    std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
    std::vector<const char*> out_names;
    for (size_t i = 0; i < num_outputs; ++i) {
        out_name_ptrs.push_back(_ort_session->GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_name_ptrs.back().get());
    }

    auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                                  out_names.data(), out_names.size());

    // Output 1: cad_descriptors (M, D)
    {
        auto shape = outs[1].GetTensorTypeAndShapeInfo().GetShape();
        _num_cad_samples = static_cast<size_t>(shape[0]);
        _desc_dim = static_cast<size_t>(shape[1]);
        const float* ptr = outs[1].GetTensorData<float>();
        _cad_descriptors.assign(ptr, ptr + _num_cad_samples * _desc_dim);
    }

    // Output 2: cad_centroids (M, 3)
    {
        auto shape = outs[2].GetTensorTypeAndShapeInfo().GetShape();
        size_t m = static_cast<size_t>(shape[0]);
        const float* ptr = outs[2].GetTensorData<float>();
        _cad_centroids.resize(m);
        for (size_t i = 0; i < m; ++i) {
            _cad_centroids[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
        }
    }

    // Output 3 & 4: cad_surface_pts & cad_surface_normals (N_surf, 3)
    {
        auto shape = outs[3].GetTensorTypeAndShapeInfo().GetShape();
        size_t n_surf = static_cast<size_t>(shape[0]);
        const float* p_ptr = outs[3].GetTensorData<float>();
        const float* n_ptr = outs[4].GetTensorData<float>();
        _mesh_pts.resize(n_surf);
        _mesh_normals.resize(n_surf);
        for (size_t i = 0; i < n_surf; ++i) {
            _mesh_pts[i] = cv::Vec3f(p_ptr[i * 3 + 0], p_ptr[i * 3 + 1], p_ptr[i * 3 + 2]);
            _mesh_normals[i] = cv::Vec3f(n_ptr[i * 3 + 0], n_ptr[i * 3 + 1], n_ptr[i * 3 + 2]);
        }
    }

    // Output 5: cad_bbox_bounds (2, 3)
    {
        const float* b_ptr = outs[5].GetTensorData<float>();
        _cad_min = cv::Vec3d(b_ptr[0], b_ptr[1], b_ptr[2]);
        _cad_max = cv::Vec3d(b_ptr[3], b_ptr[4], b_ptr[5]);
    }

    // Output 6: cad_wireframe_edges (N_edges, 2, 3)
    if (num_outputs > 6) {
        auto shape = outs[6].GetTensorTypeAndShapeInfo().GetShape();
        size_t n_edges = static_cast<size_t>(shape[0]);
        const float* e_ptr = outs[6].GetTensorData<float>();
        _cad_feature_edges.clear();
        _cad_feature_edges.reserve(n_edges);
        for (size_t i = 0; i < n_edges; ++i) {
            cv::Vec3f p0(e_ptr[i * 6 + 0], e_ptr[i * 6 + 1], e_ptr[i * 6 + 2]);
            cv::Vec3f p1(e_ptr[i * 6 + 3], e_ptr[i * 6 + 4], e_ptr[i * 6 + 5]);
            float len = static_cast<float>(cv::norm(p1 - p0));
            if (len > 1e-4f) {
                _cad_feature_edges.push_back(FeatureEdge3D{p0, p1, {0, 0, 1}, {0, 0, 1}, len});
            }
        }
    }

    // Output 10: object_meta_flags [is_floor_standing, real_diag_m, ransac_thresh_m, icp_max_corr_m]
    if (num_outputs > 10) {
        auto count = outs[10].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count >= 2) {
            const float* m_ptr = outs[10].GetTensorData<float>();
            _is_floor_standing = (m_ptr[0] > 0.5f);
            _real_bounding_diag_m = static_cast<double>(m_ptr[1]);
        }
    }
    return true;
}

bool UniversalPoseEstimator::unpackBlackBoxOnnxMetadata()
{
    std::string upper_name = _model_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    if (upper_name.find("TABLE") != std::string::npos ||
        upper_name.find("PALLET") != std::string::npos ||
        upper_name.find("RACK") != std::string::npos) {
        _is_floor_standing = true;
    }

    _half_w = 0.5 * (_cad_max[0] - _cad_min[0]);
    _half_l = 0.5 * (_cad_max[1] - _cad_min[1]);
    _bot_z  = _cad_min[2];
    _top_z  = _cad_max[2];
    _real_bounding_diag_m = std::max(0.05, cv::norm(_cad_max - _cad_min));

    float length_m = static_cast<float>(std::max(_cad_max[0] - _cad_min[0], _cad_max[1] - _cad_min[1]));
    float width_m  = static_cast<float>(std::min(_cad_max[0] - _cad_min[0], _cad_max[1] - _cad_min[1]));
    float height_m = static_cast<float>(_cad_max[2] - _cad_min[2]);

    _scene_segmenter.setExpectedObjectDimensions(length_m, width_m, height_m, _is_floor_standing);
    _scene_segmenter.setMultiInstanceMode(!_is_floor_standing, 4);
    return true;
}

bool UniversalPoseEstimator::loadUniversalProfile(const std::string& profile_json_path)
{
    if (!_scene_segmenter.loadProfileConfig(profile_json_path)) return false;
    std::ifstream ifs(profile_json_path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();
    _is_floor_standing = (content.find("FLOOR_STANDING") != std::string::npos);
    _enable_2d_frame_anchor = (content.find("\"enable_2d_frame_anchor\": true") != std::string::npos);
    return true;
}

bool UniversalPoseEstimator::isInTargetWorkspace(const cv::Vec3d& t) const
{
    if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) return false;

    if (_is_floor_standing) {
        return (t[0] >= -1.35 && t[0] <= 1.35) &&
               (t[1] >= -0.95 && t[1] <= 0.65) &&
               (t[2] >= 0.50 && t[2] <= 2.85);
    }

    double max_z = std::clamp(_real_bounding_diag_m * 8.0, 1.80, 3.20);
    return (std::abs(t[0]) <= 1.50) &&
           (std::abs(t[1]) <= 1.20) &&
           (t[2] >= 0.20 && t[2] <= max_z);
}

bool UniversalPoseEstimator::isCandidateRotationValid(const cv::Matx33d& R_candidate) const
{
    if (_is_floor_standing) {
        if (R_candidate(1, 2) > -0.50 ||
            std::abs(R_candidate(0, 2)) > 0.35 ||
            std::abs(R_candidate(2, 2)) > 0.75) {
            return false;
        }
    }
    return true;
}

bool UniversalPoseEstimator::runInference(const std::vector<cv::Vec3f>& obs_pts,
                                          std::vector<float>& out_obs_desc) const
{
    if (!_ort_loaded || !_ort_session || obs_pts.empty()) return false;

    std::vector<float> input_buf(_num_obs_points * 3, 0.0f);
    for (size_t i = 0; i < _num_obs_points; ++i) {
        const auto& p = obs_pts[i % obs_pts.size()];
        input_buf[i * 3 + 0] = p[0];
        input_buf[i * 3 + 1] = p[1];
        input_buf[i * 3 + 2] = p[2];
    }

    std::array<int64_t, 3> in_shape{1, static_cast<int64_t>(_num_obs_points), 3};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        _ort_mem_info, input_buf.data(), input_buf.size(), in_shape.data(), in_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = _ort_session->GetInputNameAllocated(0, alloc);
    auto out_name = _ort_session->GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
    const float* ptr = outs[0].GetTensorData<float>();
    out_obs_desc.assign(ptr, ptr + _num_obs_points * _desc_dim);
    return true;
}

ObjectPose UniversalPoseEstimator::solveRansacKabsch(const std::vector<cv::Vec3f>& obs_pts,
                                                     const std::vector<float>& obs_desc,
                                                     const cv::Vec4d& floor_plane,
                                                     bool has_floor) const
{
    ObjectPose best_pose;
    best_pose.model_name = _model_name;
    if (obs_pts.size() < 16 || _cad_centroids.empty() || obs_desc.empty()) return best_pose;

    const size_t N = std::min(obs_pts.size(), _num_obs_points);
    const size_t M = _cad_centroids.size();

    // Cosine similarity matching (obs_i -> cad_j)
    std::vector<int> matches(N, 0);
    for (size_t i = 0; i < N; ++i) {
        const float* o_vec = &obs_desc[i * _desc_dim];
        float best_sim = -2.0f;
        int best_j = 0;
        for (size_t j = 0; j < M; ++j) {
            const float* c_vec = &_cad_descriptors[j * _desc_dim];
            float dot = 0.0f;
            for (size_t d = 0; d < _desc_dim; ++d) dot += o_vec[d] * c_vec[d];
            if (dot > best_sim) {
                best_sim = dot;
                best_j = static_cast<int>(j);
            }
        }
        matches[i] = best_j;
    }

    std::mt19937 rng(1337);
    std::uniform_int_distribution<size_t> dist(0, N - 1);
    const double inlier_thresh = std::clamp(_real_bounding_diag_m * 0.035, 0.008, 0.045);

    for (int iter = 0; iter < 220; ++iter) {
        size_t i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        cv::Vec3d p0(obs_pts[i0]), p1(obs_pts[i1]), p2(obs_pts[i2]);
        cv::Vec3d q0(_cad_centroids[matches[i0]]), q1(_cad_centroids[matches[i1]]), q2(_cad_centroids[matches[i2]]);

        cv::Vec3d p_mean = (p0 + p1 + p2) * (1.0 / 3.0);
        cv::Vec3d q_mean = (q0 + q1 + q2) * (1.0 / 3.0);

        cv::Matx33d H = cv::Matx33d::zeros();
        const cv::Vec3d dp[3] = {p0 - p_mean, p1 - p_mean, p2 - p_mean};
        const cv::Vec3d dq[3] = {q0 - q_mean, q1 - q_mean, q2 - q_mean};
        for (int k = 0; k < 3; ++k)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    H(r, c) += dq[k][r] * dp[k][c];

        cv::Mat U, S, Vt;
        cv::SVD::compute(cv::Mat(H), S, U, Vt);
        cv::Mat R_mat = Vt.t() * U.t();
        if (cv::determinant(R_mat) < 0) {
            Vt.row(2) *= -1.0;
            R_mat = Vt.t() * U.t();
        }

        cv::Matx33d R_cand(R_mat);
        if (!isCandidateRotationValid(R_cand)) continue;

        cv::Vec3d t_cand = p_mean - R_cand * q_mean;
        if (!isInTargetWorkspace(t_cand)) continue;

        int inliers = 0;
        double err_sum = 0.0;
        for (size_t i = 0; i < N; i += 2) {
            cv::Vec3d pred = R_cand * cv::Vec3d(_cad_centroids[matches[i]]) + t_cand;
            double e = cv::norm(pred - cv::Vec3d(obs_pts[i]));
            if (e <= inlier_thresh) {
                ++inliers;
                err_sum += e * e;
            }
        }

        if (inliers > best_pose.inliers_count) {
            best_pose.valid = true;
            best_pose.inliers_count = inliers;
            best_pose.R = R_cand;
            best_pose.t = t_cand;
            best_pose.rmse_m = std::sqrt(err_sum / std::max(1, inliers));
            best_pose.confidence = std::clamp(static_cast<float>(inliers) / static_cast<float>(N / 2), 0.15f, 0.99f);
        }
    }

    if (best_pose.valid && _is_floor_standing && has_floor) {
        // Align CAD +Z upright with measured floor plane normal so rear legs never float
        cv::Vec3d z_axis(0.0, -0.9885, 0.1510);
        if (floor_plane[1] < -0.82 && floor_plane[2] >= 0.08 && floor_plane[2] <= 0.25) {
            z_axis = cv::Vec3d(floor_plane[0] * 0.35, floor_plane[1], floor_plane[2]);
        }
        z_axis *= (1.0 / cv::norm(z_axis));
        cv::Vec3d y_axis(best_pose.R(0, 1), best_pose.R(1, 1), best_pose.R(2, 1));
        y_axis -= z_axis * z_axis.dot(y_axis);
        y_axis *= (1.0 / std::max(1e-6, cv::norm(y_axis)));
        cv::Vec3d x_axis = y_axis.cross(z_axis);
        best_pose.R = cv::Matx33d(
            x_axis[0], y_axis[0], z_axis[0],
            x_axis[1], y_axis[1], z_axis[1],
            x_axis[2], y_axis[2], z_axis[2]);
    }

    if (best_pose.valid) {
        best_pose.inlier_ratio = static_cast<double>(best_pose.inliers_count) / static_cast<double>(std::max<size_t>(1, N / 2));
        best_pose.t_glass = best_pose.t;
        cv::Vec3d euler = rotationMatrixToEulerDeg(best_pose.R);
        best_pose.roll_deg = euler[0];
        best_pose.pitch_deg = euler[1];
        best_pose.yaw_deg = euler[2];
        best_pose.method = "universal_ransac_kabsch";
    }
    return best_pose;
}

void UniversalPoseEstimator::refineMultiScaleMeshICP(const std::vector<cv::Vec3f>& scene_pts,
                                                     ObjectPose& pose,
                                                     const CameraIntrinsics& intr) const
{
    if (!pose.valid || _mesh_pts.empty() || scene_pts.size() < 16) return;
    (void)intr;

    const double radii[2] = {_real_bounding_diag_m * 0.04, _real_bounding_diag_m * 0.02};
    for (double max_corr : radii) {
        cv::Vec3d t_shift(0.0, 0.0, 0.0);
        int count = 0;

        for (size_t i = 0; i < scene_pts.size(); i += 4) {
            cv::Vec3d p_obs(scene_pts[i]);
            double best_d = max_corr;
            cv::Vec3d best_n(0, 0, 0), best_q(0, 0, 0);

            for (size_t j = 0; j < _mesh_pts.size(); j += 8) {
                cv::Vec3d q_cam = pose.R * cv::Vec3d(_mesh_pts[j]) + pose.t;
                cv::Vec3d n_cam = pose.R * cv::Vec3d(_mesh_normals[j]);
                // Visibility filter: only match front-facing CAD triangles (view_dot < 0)
                if (n_cam.dot(q_cam) >= 0.0) continue;

                double d = cv::norm(p_obs - q_cam);
                if (d < best_d) {
                    best_d = d;
                    best_q = q_cam;
                    best_n = n_cam;
                }
            }
            if (best_d < max_corr) {
                double proj = (p_obs - best_q).dot(best_n);
                t_shift += best_n * proj;
                ++count;
            }
        }
        if (count >= 10) {
            pose.t += (t_shift * (0.65 / count));
        }
    }
}

void UniversalPoseEstimator::refinePoseUniversalCrease2D3D(ObjectPose& pose,
                                                           const cv::Mat& rgb,
                                                           const CameraIntrinsics& intr) const
{
    if (!pose.valid || rgb.empty() || _cad_feature_edges.empty()) return;

    cv::Mat gray, edges, dt;
    if (rgb.channels() == 3) cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    else gray = rgb;

    cv::Canny(gray, edges, 40, 120);
    cv::Mat inv_edges = 255 - edges;
    cv::distanceTransform(inv_edges, dt, cv::DIST_L2, 3);

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : (0.5 * rgb.cols);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : (0.5 * rgb.rows);

    auto evalChamfer = [&](const cv::Matx33d& R_try, const cv::Vec3d& t_try) -> double {
        double sum = 0.0;
        int valid = 0;
        for (const auto& e : _cad_feature_edges) {
            for (float a : {0.2f, 0.5f, 0.8f}) {
                cv::Vec3f p_local = e.p0 * (1.0f - a) + e.p1 * a;
                cv::Vec3d p_cam = R_try * cv::Vec3d(p_local) + t_try;
                if (p_cam[2] <= 0.15) continue;
                int u = static_cast<int>(std::round(fx * p_cam[0] / p_cam[2] + cx));
                int v = static_cast<int>(std::round(fy * p_cam[1] / p_cam[2] + cy));
                if (u >= 0 && u < dt.cols && v >= 0 && v < dt.rows) {
                    sum += std::min(25.0f, dt.at<float>(v, u));
                    ++valid;
                }
            }
        }
        return (valid > 8) ? (sum / valid) : 1e6;
    };

    double best_cost = evalChamfer(pose.R, pose.t);
    for (double dz : {-0.015, -0.007, 0.007, 0.015}) {
        for (double dx : {-0.010, 0.0, 0.010}) {
            cv::Vec3d t_try = pose.t + cv::Vec3d(dx, 0.0, dz);
            double c = evalChamfer(pose.R, t_try);
            if (c < best_cost) {
                best_cost = c;
                pose.t = t_try;
            }
        }
    }
}

ObjectPose UniversalPoseEstimator::estimate(const cv::Mat& rgb,
                                            const cv::Mat& depth_z16,
                                            const CameraIntrinsics& intr)
{
    ObjectPose pose;
    pose.model_name = _model_name;
    if (!_initialized || depth_z16.empty()) return pose;

    SegmentationResult seg_res = _scene_segmenter.segment(depth_z16, intr, cv::Rect(), _num_obs_points);
    if (!seg_res.valid) return pose;

    std::vector<float> obs_desc;
    if (!runInference(seg_res.object_points, obs_desc)) return pose;

    pose = solveRansacKabsch(seg_res.object_points, obs_desc, seg_res.floor_plane, seg_res.floor_detected);
    if (!pose.valid) return pose;

    refineMultiScaleMeshICP(seg_res.dense_object_points, pose, intr);
    refinePoseUniversalCrease2D3D(pose, rgb, intr);
    pose.bbox_2d = projectCadBoundingBox(pose.R, pose.t, intr, depth_z16.cols, depth_z16.rows, &pose.corners_2d);
    return pose;
}

std::vector<ObjectPose> UniversalPoseEstimator::estimateMultiObjects(const cv::Mat& rgb,
                                                                     const cv::Mat& depth_z16,
                                                                     const CameraIntrinsics& intr)
{
    std::vector<ObjectPose> results;
    if (!_initialized || depth_z16.empty()) return results;

    SegmentationResult seg_res = _scene_segmenter.segment(depth_z16, intr, cv::Rect(), _num_obs_points);
    if (!seg_res.valid) return results;

    if (_is_floor_standing || seg_res.multi_clusters.size() <= 1) {
        ObjectPose single = estimate(rgb, depth_z16, intr);
        if (single.valid) results.push_back(single);
        return results;
    }

    for (size_t i = 0; i < seg_res.multi_clusters.size(); ++i) {
        const auto& cl = seg_res.multi_clusters[i];
        if (cl.points.size() < 20) continue;

        std::vector<float> obs_desc;
        if (!runInference(cl.points, obs_desc)) continue;

        ObjectPose inst_pose = solveRansacKabsch(cl.points, obs_desc, seg_res.floor_plane, false);
        if (!inst_pose.valid) continue;

        refineMultiScaleMeshICP(cl.points, inst_pose, intr);
        refinePoseUniversalCrease2D3D(inst_pose, rgb, intr);
        inst_pose.bbox_2d = projectCadBoundingBox(inst_pose.R, inst_pose.t, intr,
                                                  depth_z16.cols, depth_z16.rows, &inst_pose.corners_2d);
        inst_pose.model_name = _model_name + " #" + std::to_string(i + 1);
        results.push_back(inst_pose);
    }
    return results;
}

cv::Rect2f UniversalPoseEstimator::projectCadBoundingBox(const cv::Matx33d& R,
                                                         const cv::Vec3d& t,
                                                         const CameraIntrinsics& intr,
                                                         int img_w,
                                                         int img_h,
                                                         std::vector<cv::Point2f>* out_corners) const
{
    (void)img_w; (void)img_h;
    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : 320.0;
    const double cy = (intr.ppy > 10.0) ? intr.ppy : 240.0;

    const double x0 = _cad_min[0], x1 = _cad_max[0];
    const double y0 = _cad_min[1], y1 = _cad_max[1];
    const double z0 = _cad_min[2], z1 = _cad_max[2];

    const cv::Vec3d corners_3d[8] = {
        {x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
        {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}
    };

    float min_u = 1e9f, min_v = 1e9f, max_u = -1e9f, max_v = -1e9f;
    if (out_corners) out_corners->clear();

    for (int i = 0; i < 8; ++i) {
        cv::Vec3d p = R * corners_3d[i] + t;
        double z = std::max(0.05, p[2]);
        float u = static_cast<float>(fx * p[0] / z + cx);
        float v = static_cast<float>(fy * p[1] / z + cy);
        if (out_corners) out_corners->emplace_back(u, v);
        min_u = std::min(min_u, u); max_u = std::max(max_u, u);
        min_v = std::min(min_v, v); max_v = std::max(max_v, v);
    }
    return cv::Rect2f(min_u, min_v, max_u - min_u, max_v - min_v);
}

void UniversalPoseEstimator::drawOverlay(cv::Mat& img,
                                         const ObjectPose& pose,
                                         const CameraIntrinsics& intr) const
{
    if (img.empty() || !pose.valid) return;

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : (0.5 * img.cols);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : (0.5 * img.rows);

    auto proj_f = [&](double x, double y, double z) -> cv::Point2f {
        cv::Vec3d p = pose.R * cv::Vec3d(x, y, z) + pose.t;
        double pz = std::max(0.05, p[2]);
        return cv::Point2f(static_cast<float>(fx * p[0] / pz + cx),
                           static_cast<float>(fy * p[1] / pz + cy));
    };
    auto to_pt = [](const cv::Point2f& p) -> cv::Point {
        return cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
    };

    const cv::Scalar CYAN(255, 229, 0), GREEN(50, 235, 50), YELLOW(0, 235, 255), RED(40, 40, 255), BLUE(255, 120, 30), WHITE(245, 245, 245);

    // 1. Project 3D CAD feature/crease wireframe edges for ANY object
    for (const auto& e : _cad_feature_edges) {
        cv::Point2f a = proj_f(e.p0[0], e.p0[1], e.p0[2]);
        cv::Point2f b = proj_f(e.p1[0], e.p1[1], e.p1[2]);
        cv::line(img, to_pt(a), to_pt(b), CYAN, 2, cv::LINE_AA);
    }

    // 2. Draw 8-corner 3D Oriented Bounding Box
    std::vector<cv::Point2f> c2d = pose.corners_2d;
    if (c2d.size() != 8) {
        projectCadBoundingBox(pose.R, pose.t, intr, img.cols, img.rows, &c2d);
    }
    if (c2d.size() == 8) {
        for (int i = 0; i < 4; ++i) {
            cv::line(img, to_pt(c2d[i]), to_pt(c2d[(i + 1) % 4]), GREEN, 2, cv::LINE_AA);
            cv::line(img, to_pt(c2d[i + 4]), to_pt(c2d[((i + 1) % 4) + 4]), CYAN, 2, cv::LINE_AA);
            cv::line(img, to_pt(c2d[i]), to_pt(c2d[i + 4]), YELLOW, 2, cv::LINE_AA);
        }
    }

    // 3. Draw 3D Coordinate Axes at object center
    cv::Point2f p_c = proj_f(0, 0, 0);
    cv::Point2f p_x = proj_f(0.10, 0, 0);
    cv::Point2f p_y = proj_f(0, 0.10, 0);
    cv::Point2f p_z = proj_f(0, 0, 0.10);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_x), RED, 2, cv::LINE_AA, 0, 0.22);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_y), GREEN, 2, cv::LINE_AA, 0, 0.22);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_z), BLUE, 2, cv::LINE_AA, 0, 0.22);

    // 4. Universal Object Telemetry Label
    std::ostringstream oss;
    oss << "[" << pose.model_name << "] Z:" << std::fixed << std::setprecision(2) << pose.t[2]
        << "m | Yaw:" << std::setprecision(1) << pose.yaw_deg << " deg";
    cv::Point lbl_pt = (c2d.size() == 8) ? to_pt(c2d[4]) : to_pt(p_c);
    lbl_pt.y = std::max(24, lbl_pt.y - 10);
    cv::putText(img, oss.str(), lbl_pt, cv::FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv::LINE_AA);
}

void UniversalPoseEstimator::drawMultiOverlay(cv::Mat& img,
                                              const std::vector<ObjectPose>& poses,
                                              const CameraIntrinsics& intr) const
{
    for (const auto& p : poses) {
        drawOverlay(img, p, intr);
    }
}

void UniversalPoseEstimator::transformPoseCameraFrame(ObjectPose& pose,
                                                      const cv::Matx33d& target_from_source_R,
                                                      const cv::Vec3d& target_from_source_t,
                                                      const CameraIntrinsics& target_intr,
                                                      int target_width,
                                                      int target_height) const
{
    pose.R = target_from_source_R * pose.R;
    pose.t = target_from_source_R * pose.t + target_from_source_t;
    pose.t_glass = pose.t;
    pose.bbox_2d = projectCadBoundingBox(pose.R, pose.t, target_intr,
                                         target_width, target_height, &pose.corners_2d);
}

cv::Vec3d UniversalPoseEstimator::rotationMatrixToEulerDeg(const cv::Matx33d& R)
{
    cv::Vec3d v_up(R(0, 2), R(1, 2), R(2, 2));
    double roll = std::asin(std::clamp(v_up[0], -1.0, 1.0)) * 180.0 / kPi;
    double pitch = std::asin(std::clamp(v_up[2], -1.0, 1.0)) * 180.0 / kPi;
    double yaw = std::atan2(-R(0, 1), R(0, 0)) * 180.0 / kPi;
    return cv::Vec3d(roll, pitch, yaw);
}

} // namespace tmcad

