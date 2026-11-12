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
#include <unordered_map>

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
        _loaded_onnx_path = onnx_path;
        _model_name = extractStem(onnx_path);

        if (_model_name == "model_obs_npu" || _model_name == "model_obs_backbone") {
            auto parent = std::filesystem::path(onnx_path).parent_path().filename().string();
            if (!parent.empty() && parent != "models") _model_name = parent;
        }

        // Reset cached CAD buffers before unpacking new model
        _cad_descriptors.clear();
        _cad_centroids.clear();
        _mesh_pts.clear();
        _mesh_normals.clear();
        _cad_feature_edges.clear();
        _cad_top_polygon.clear();
        _cad_foot_centers.clear();
        _meta_unpacked = false;
        _symmetry_code = 0;
        _is_floor_standing = false;
        _enable_2d_frame_anchor = false;
        _initialized = false;

        if (unpackBlackBoxOnnx()) {
            unpackBlackBoxOnnxMetadata();

            // Auto-discover optional <CAD_NAME>_profile.json for adaptive solver hints
            std::filesystem::path onnx_p(onnx_path);
            std::vector<std::string> profile_cands = {
                (onnx_p.parent_path() / (_model_name + "_profile.json")).string(),
                "CAD_DATA/" + _model_name + "/" + _model_name + "_profile.json",
                "configs/" + _model_name + "_profile.json"
            };
            for (const auto& prof : profile_cands) {
                if (fileExists(prof) && loadUniversalProfile(prof)) {
                    break;
                }
            }

            if (_symmetry_code == 0) {
                std::vector<std::string> sym_cands = {
                    (onnx_p.parent_path() / (_model_name + "_symmetry.json")).string(),
                    "CAD_DATA/" + _model_name + "/" + _model_name + "_symmetry.json"
                };
                for (const auto& sym_p : sym_cands) {
                    if (fileExists(sym_p)) {
                        std::ifstream sifs(sym_p);
                        std::stringstream sss;
                        sss << sifs.rdbuf();
                        const std::string sc = sss.str();
                        if (sc.find("C4") != std::string::npos || sc.find("D2") != std::string::npos) {
                            _symmetry_code = 2;
                        } else if (sc.find("C2") != std::string::npos || sc.find("\"is_symmetric\": true") != std::string::npos) {
                            _symmetry_code = 1;
                        }
                        break;
                    }
                }
            }

            _initialized = true;
            std::cout << "[UniversalPoseEstimator] Ready with Black-Box ONNX (" << _model_name
                      << ") | Mode: " << (_is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF")
                      << " | Diag: " << _real_bounding_diag_m << " m"
                      << " | CAD Pts: " << _cad_centroids.size()
                      << " | Surf Pts: " << _mesh_pts.size()
                      << " | Edges: " << _cad_feature_edges.size() << std::endl;
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
    const size_t num_outputs = _ort_session->GetOutputCount();
    if (num_outputs < 5) return false;

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr = _ort_session->GetInputNameAllocated(0, alloc);
    const char* in_names[] = {in_name_ptr.get()};

    // Map all available ONNX outputs by exact tensor name (schema-agnostic & future-proof)
    std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
    std::vector<const char*> out_names;
    std::unordered_map<std::string, size_t> name_to_idx;
    out_name_ptrs.reserve(num_outputs);
    out_names.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        out_name_ptrs.push_back(_ort_session->GetOutputNameAllocated(i, alloc));
        const char* name_cstr = out_name_ptrs.back().get();
        out_names.push_back(name_cstr);
        name_to_idx[std::string(name_cstr)] = i;
    }

    // Determine expected observation point count from ONNX input shape if fixed
    {
        auto in_shape = _ort_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() == 3 && in_shape[1] > 0) {
            _num_obs_points = static_cast<size_t>(in_shape[1]);
        }
    }

    std::vector<float> dummy_in(_num_obs_points * 3, 0.0f);
    std::array<int64_t, 3> in_shape{1, static_cast<int64_t>(_num_obs_points), 3};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        _ort_mem_info, dummy_in.data(), dummy_in.size(), in_shape.data(), in_shape.size());

    auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                                  out_names.data(), out_names.size());

    auto findOut = [&](const std::string& key) -> const Ort::Value* {
        auto it = name_to_idx.find(key);
        if (it != name_to_idx.end() && it->second < outs.size()) {
            return &outs[it->second];
        }
        return nullptr;
    };

    // 1. cad_descriptors: (M, D)
    if (const Ort::Value* v = findOut("cad_descriptors")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[0] > 0 && shape[1] > 0 && ptr) {
            _num_cad_samples = static_cast<size_t>(shape[0]);
            _desc_dim = static_cast<size_t>(shape[1]);
            _cad_descriptors.assign(ptr, ptr + _num_cad_samples * _desc_dim);
        }
    }

    // 2. cad_centroids: (M, 3)
    if (const Ort::Value* v = findOut("cad_centroids")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[1] == 3 && ptr) {
            size_t m = static_cast<size_t>(shape[0]);
            _cad_centroids.resize(m);
            for (size_t i = 0; i < m; ++i) {
                _cad_centroids[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
            }
        }
    }

    // 3. cad_surface_pts & cad_surface_normals: (N_surf, 3)
    if (const Ort::Value* vp = findOut("cad_surface_pts")) {
        auto shape_p = vp->GetTensorTypeAndShapeInfo().GetShape();
        const float* p_ptr = vp->GetTensorData<float>();
        const Ort::Value* vn = findOut("cad_surface_normals");
        const float* n_ptr = vn ? vn->GetTensorData<float>() : nullptr;
        if (shape_p.size() == 2 && shape_p[1] == 3 && p_ptr) {
            size_t n_surf = static_cast<size_t>(shape_p[0]);
            _mesh_pts.resize(n_surf);
            _mesh_normals.resize(n_surf);
            for (size_t i = 0; i < n_surf; ++i) {
                _mesh_pts[i] = cv::Vec3f(p_ptr[i * 3 + 0], p_ptr[i * 3 + 1], p_ptr[i * 3 + 2]);
                if (n_ptr) {
                    _mesh_normals[i] = cv::Vec3f(n_ptr[i * 3 + 0], n_ptr[i * 3 + 1], n_ptr[i * 3 + 2]);
                } else {
                    _mesh_normals[i] = cv::Vec3f(0.0f, 0.0f, 1.0f);
                }
            }
        }
    }

    // 4. cad_feature_edges (E, 13) [p0(3), p1(3), n0(3), n1(3), len] or cad_wireframe_edges (E, 6)
    if (const Ort::Value* v = findOut("cad_feature_edges")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* e_ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && e_ptr) {
            size_t n_edges = static_cast<size_t>(shape[0]);
            size_t cols = static_cast<size_t>(shape[1]);
            _cad_feature_edges.clear();
            _cad_feature_edges.reserve(n_edges);
            if (cols >= 13) {
                for (size_t i = 0; i < n_edges; ++i) {
                    const float* r = e_ptr + i * cols;
                    cv::Vec3f p0(r[0], r[1], r[2]);
                    cv::Vec3f p1(r[3], r[4], r[5]);
                    cv::Vec3f n0(r[6], r[7], r[8]);
                    cv::Vec3f n1(r[9], r[10], r[11]);
                    float len = r[12];
                    if (len <= 1e-5f) len = static_cast<float>(cv::norm(p1 - p0));
                    if (len > 1e-4f) {
                        _cad_feature_edges.push_back(FeatureEdge3D{p0, p1, n0, n1, len});
                    }
                }
            } else if (cols >= 6) {
                for (size_t i = 0; i < n_edges; ++i) {
                    const float* r = e_ptr + i * cols;
                    cv::Vec3f p0(r[0], r[1], r[2]);
                    cv::Vec3f p1(r[3], r[4], r[5]);
                    float len = static_cast<float>(cv::norm(p1 - p0));
                    if (len > 1e-4f) {
                        _cad_feature_edges.push_back(FeatureEdge3D{p0, p1, {0, 0, 1}, {0, 0, 1}, len});
                    }
                }
            }
        }
    } else if (const Ort::Value* vw = findOut("cad_wireframe_edges")) {
        auto shape = vw->GetTensorTypeAndShapeInfo().GetShape();
        const float* e_ptr = vw->GetTensorData<float>();
        if (shape.size() >= 2 && e_ptr) {
            size_t n_edges = static_cast<size_t>(shape[0]);
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
    }

    // 5. cad_top_poly & cad_foot_centers
    if (const Ort::Value* v = findOut("cad_top_poly")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[1] == 3 && ptr) {
            size_t n = static_cast<size_t>(shape[0]);
            _cad_top_polygon.resize(n);
            for (size_t i = 0; i < n; ++i) {
                _cad_top_polygon[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
            }
        }
    }
    if (const Ort::Value* v = findOut("cad_foot_centers")) {
        auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
        const float* ptr = v->GetTensorData<float>();
        if (shape.size() == 2 && shape[1] == 3 && ptr) {
            size_t n = static_cast<size_t>(shape[0]);
            _cad_foot_centers.resize(n);
            for (size_t i = 0; i < n; ++i) {
                _cad_foot_centers[i] = cv::Vec3f(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
            }
        }
    }

    // 6. cad_meta (16,) from export_hierarchy_onnx.py (canonical schema)
    if (const Ort::Value* v = findOut("cad_meta")) {
        auto count = v->GetTensorTypeAndShapeInfo().GetElementCount();
        const float* ptr = v->GetTensorData<float>();
        if (count >= 12 && ptr) {
            _cad_min = cv::Vec3d(ptr[0], ptr[1], ptr[2]);
            _cad_max = cv::Vec3d(ptr[3], ptr[4], ptr[5]);
            _half_w  = static_cast<double>(ptr[6]);
            _half_l  = static_cast<double>(ptr[7]);
            _top_z   = static_cast<double>(ptr[8]);
            _bot_z   = static_cast<double>(ptr[9]);
            _symmetry_code = static_cast<int>(std::round(ptr[10]));
            _is_floor_standing = (ptr[11] > 0.5f);
            _enable_2d_frame_anchor = _is_floor_standing;
            _meta_unpacked = true;
        }
    } else {
        // Optional fallback schema: cad_bbox_bounds + object_meta_flags
        if (const Ort::Value* vb = findOut("cad_bbox_bounds")) {
            auto count = vb->GetTensorTypeAndShapeInfo().GetElementCount();
            const float* b_ptr = vb->GetTensorData<float>();
            if (count >= 6 && b_ptr) {
                _cad_min = cv::Vec3d(b_ptr[0], b_ptr[1], b_ptr[2]);
                _cad_max = cv::Vec3d(b_ptr[3], b_ptr[4], b_ptr[5]);
                _meta_unpacked = true;
            }
        }
        if (const Ort::Value* vm = findOut("object_meta_flags")) {
            auto count = vm->GetTensorTypeAndShapeInfo().GetElementCount();
            const float* m_ptr = vm->GetTensorData<float>();
            if (count >= 2 && m_ptr) {
                _is_floor_standing = (m_ptr[0] > 0.5f);
                _real_bounding_diag_m = static_cast<double>(m_ptr[1]);
                _meta_unpacked = true;
            }
        }
    }

    return (!_cad_descriptors.empty() && !_cad_centroids.empty());
}

bool UniversalPoseEstimator::unpackBlackBoxOnnxMetadata()
{
    if (!_meta_unpacked) {
        std::string upper_name = _model_name;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

        if (upper_name.find("TABLE") != std::string::npos ||
            upper_name.find("PALLET") != std::string::npos ||
            upper_name.find("RACK") != std::string::npos) {
            _is_floor_standing = true;
        }
    }

    // Fallback bbox calculation from mesh/centroids if cad_meta wasn't present
    if (!_meta_unpacked && !_mesh_pts.empty()) {
        cv::Vec3d bmin(1e9, 1e9, 1e9), bmax(-1e9, -1e9, -1e9);
        for (const auto& p : _mesh_pts) {
            for (int k = 0; k < 3; ++k) {
                bmin[k] = std::min(bmin[k], static_cast<double>(p[k]));
                bmax[k] = std::max(bmax[k], static_cast<double>(p[k]));
            }
        }
        _cad_min = bmin;
        _cad_max = bmax;
    }

    _half_w = 0.5 * (_cad_max[0] - _cad_min[0]);
    _half_l = 0.5 * (_cad_max[1] - _cad_min[1]);
    _bot_z  = _cad_min[2];
    _top_z  = _cad_max[2];
    _real_bounding_diag_m = std::max(0.05, cv::norm(_cad_max - _cad_min));

    float length_m = static_cast<float>(std::max(_cad_max[0] - _cad_min[0], _cad_max[1] - _cad_min[1]));
    float width_m  = static_cast<float>(std::min(_cad_max[0] - _cad_min[0], _cad_max[1] - _cad_min[1]));
    float height_m = static_cast<float>(_cad_max[2] - _cad_min[2]);

    _topology_type = _is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF";
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
    _is_floor_standing = (content.find("\"placement_mode\": \"FLOOR_STANDING\"") != std::string::npos ||
                          content.find("FRAME_HOLLOW") != std::string::npos);
    _enable_2d_frame_anchor = (content.find("\"enable_2d_frame_anchor\": true") != std::string::npos);
    _topology_type = _is_floor_standing ? "FLOOR_STANDING" : "FREE_6DOF";
    _scene_segmenter.setMultiInstanceMode(!_is_floor_standing, 4);
    return true;
}

bool UniversalPoseEstimator::isInTargetWorkspace(const cv::Vec3d& t) const
{
    if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) return false;

    if (_is_floor_standing) {
        return (t[0] >= -1.35 && t[0] <= 1.35) &&
               (t[1] >= -0.95 && t[1] <= 0.65) &&
               (t[2] >= 0.45 && t[2] <= 2.95);
    }

    double max_z = std::clamp(_real_bounding_diag_m * 8.0, 1.80, 3.20);
    return (std::abs(t[0]) <= 1.50) &&
           (std::abs(t[1]) <= 1.20) &&
           (t[2] >= 0.15 && t[2] <= max_z);
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
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {"D_obs"};

    try {
        auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
        if (outs.empty()) return false;
        const float* ptr = outs[0].GetTensorData<float>();
        if (!ptr) return false;
        out_obs_desc.assign(ptr, ptr + _num_obs_points * _desc_dim);
        return true;
    } catch (...) {
        auto out0_name = _ort_session->GetOutputNameAllocated(0, alloc);
        const char* fallback_names[] = {out0_name.get()};
        auto outs = _ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, fallback_names, 1);
        if (outs.empty()) return false;
        const float* ptr = outs[0].GetTensorData<float>();
        if (!ptr) return false;
        out_obs_desc.assign(ptr, ptr + _num_obs_points * _desc_dim);
        return true;
    }
}

ObjectPose UniversalPoseEstimator::solveRansacKabsch(const std::vector<cv::Vec3f>& obs_pts,
                                                     const std::vector<float>& obs_desc,
                                                     const cv::Vec4d& floor_plane,
                                                     bool has_floor) const
{
    ObjectPose best_pose;
    best_pose.model_name = _model_name;
    if (obs_pts.size() < 16 || _cad_centroids.empty() || obs_desc.empty() || _cad_descriptors.empty()) {
        return best_pose;
    }

    const size_t N = std::min(obs_pts.size(), _num_obs_points);
    const size_t M = _cad_centroids.size();
    if (_cad_descriptors.size() < M * _desc_dim || obs_desc.size() < N * _desc_dim) {
        return best_pose;
    }

    // 1. Cosine similarity matching (obs_i -> cad_j)
    std::vector<int> matches(N, 0);
    std::vector<float> top_scores(N, 0.0f);
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
        top_scores[i] = best_sim;
    }

    // 2. Cluster NMS: keep top 3 observation points per CAD centroid so symmetric modes stay balanced
    std::unordered_map<int, std::vector<int>> centroid_to_obs;
    centroid_to_obs.reserve(M);
    for (size_t i = 0; i < N; ++i) {
        centroid_to_obs[matches[i]].push_back(static_cast<int>(i));
    }

    std::vector<int> nms_indices;
    nms_indices.reserve(N);
    for (auto& kv : centroid_to_obs) {
        auto& indices = kv.second;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return top_scores[a] > top_scores[b];
        });
        int take = std::min(3, static_cast<int>(indices.size()));
        for (int k = 0; k < take; ++k) {
            nms_indices.push_back(indices[k]);
        }
    }
    if (nms_indices.size() < 16) {
        nms_indices.resize(N);
        for (size_t i = 0; i < N; ++i) nms_indices[i] = static_cast<int>(i);
    }

    const double inlier_thresh = std::clamp(_real_bounding_diag_m * 0.038, 0.008, 0.045);
    int best_inliers = 0;
    double best_err_sum = 0.0;
    std::vector<uint8_t> best_mask(N, 0);

    cv::Vec3d n_fl(floor_plane[0], floor_plane[1], floor_plane[2]);
    double n_fl_norm = cv::norm(n_fl);
    bool valid_floor = has_floor && (n_fl_norm > 1e-6);
    if (valid_floor) {
        n_fl *= (1.0 / n_fl_norm);
        if (n_fl[1] > 0.0) n_fl = -n_fl;
    }

    // 3. Deterministic Floor-Aligned PCA Hypothesis (for FLOOR_STANDING frames like TABLE-DEMO)
    if (_is_floor_standing && valid_floor) {
        cv::Vec3d mu_obs_all(0.0, 0.0, 0.0), mu_cad_all(0.0, 0.0, 0.0);
        for (size_t i = 0; i < N; ++i) {
            mu_obs_all += cv::Vec3d(obs_pts[i]);
            mu_cad_all += cv::Vec3d(_cad_centroids[matches[i]]);
        }
        mu_obs_all *= (1.0 / static_cast<double>(N));
        mu_cad_all *= (1.0 / static_cast<double>(N));

        cv::Matx33d cov = cv::Matx33d::zeros();
        for (size_t i = 0; i < N; ++i) {
            cv::Vec3d d = cv::Vec3d(obs_pts[i]) - mu_obs_all;
            cov += d * d.t();
        }
        cv::Matx33d U_cov, Vt_cov;
        cv::Vec3d W_cov;
        cv::SVD::compute(cov, W_cov, U_cov, Vt_cov);
        cv::Vec3d v_beam(Vt_cov(0, 0), Vt_cov(0, 1), Vt_cov(0, 2));
        if (v_beam[0] < 0.0) v_beam = -v_beam;

        cv::Vec3d cad_y = v_beam - v_beam.dot(n_fl) * n_fl;
        double y_norm = cv::norm(cad_y);
        if (y_norm > 1e-6) {
            cad_y *= (1.0 / y_norm);
            cv::Vec3d cad_z = n_fl;
            cv::Vec3d cad_x = cad_y.cross(cad_z);
            double x_norm = cv::norm(cad_x);
            if (x_norm > 1e-6) {
                cad_x *= (1.0 / x_norm);
                if (cad_x[2] > 0.0) { cad_x = -cad_x; cad_y = -cad_y; }

                cv::Matx33d R_fl(
                    cad_x[0], cad_y[0], cad_z[0],
                    cad_x[1], cad_y[1], cad_z[1],
                    cad_x[2], cad_y[2], cad_z[2]
                );
                cv::Matx33d R_flip(-1.0,  0.0, 0.0,
                                    0.0, -1.0, 0.0,
                                    0.0,  0.0, 1.0);
                cv::Matx33d R_cands[2] = {R_fl, R_fl * R_flip};

                for (int h = 0; h < 2; ++h) {
                    const auto& R_cand = R_cands[h];
                    if (!isCandidateRotationValid(R_cand)) continue;
                    cv::Vec3d t_cand = mu_obs_all - R_cand * mu_cad_all;
                    if (!isInTargetWorkspace(t_cand)) continue;

                    int inliers = 0;
                    double err_sum = 0.0;
                    std::vector<uint8_t> mask(N, 0);
                    for (size_t i = 0; i < N; ++i) {
                        cv::Vec3d pred = R_cand * cv::Vec3d(_cad_centroids[matches[i]]) + t_cand;
                        double e = cv::norm(pred - cv::Vec3d(obs_pts[i]));
                        if (e <= inlier_thresh) {
                            ++inliers;
                            err_sum += e * e;
                            mask[i] = 1;
                        }
                    }
                    if (inliers > best_inliers) {
                        best_inliers = inliers;
                        best_err_sum = err_sum;
                        best_mask = std::move(mask);
                        best_pose.valid = true;
                        best_pose.R = R_cand;
                        best_pose.t = t_cand;
                    }
                }
            }
        }
    }

    // 4. Confidence-Weighted 4-Point RANSAC Kabsch SVD
    std::mt19937 rng(1337);
    std::vector<float> sample_weights(nms_indices.size());
    for (size_t i = 0; i < nms_indices.size(); ++i) {
        sample_weights[i] = std::pow(std::max(0.001f, top_scores[nms_indices[i]]), 1.5f);
    }
    std::discrete_distribution<int> dist(sample_weights.begin(), sample_weights.end());

    for (int iter = 0; iter < 450; ++iter) {
        int idxs[4];
        for (int s = 0; s < 4; ++s) {
            idxs[s] = nms_indices[dist(rng)];
        }
        if (idxs[0] == idxs[1] || idxs[1] == idxs[2] || idxs[0] == idxs[2] ||
            idxs[3] == idxs[0] || idxs[3] == idxs[1] || idxs[3] == idxs[2]) continue;

        cv::Vec3d p_mean(0.0, 0.0, 0.0), q_mean(0.0, 0.0, 0.0);
        for (int s = 0; s < 4; ++s) {
            p_mean += cv::Vec3d(obs_pts[idxs[s]]);
            q_mean += cv::Vec3d(_cad_centroids[matches[idxs[s]]]);
        }
        p_mean *= 0.25;
        q_mean *= 0.25;

        cv::Matx33d H = cv::Matx33d::zeros();
        for (int s = 0; s < 4; ++s) {
            cv::Vec3d dp = cv::Vec3d(obs_pts[idxs[s]]) - p_mean;
            cv::Vec3d dq = cv::Vec3d(_cad_centroids[matches[idxs[s]]]) - q_mean;
            H += dp * dq.t();
        }

        cv::Matx33d U, Vt;
        cv::Vec3d W;
        cv::SVD::compute(H, W, U, Vt);
        // Reject degenerate / collinear / single-centroid correspondence hypotheses
        if (W[0] < 1e-6 || W[1] < 1e-7) continue;
        cv::Matx33d R_cand = U * Vt;
        if (cv::determinant(R_cand) < 0.0) {
            cv::Matx33d fix = cv::Matx33d::eye();
            fix(2, 2) = -1.0;
            R_cand = U * fix * Vt;
        }

        if (!isCandidateRotationValid(R_cand)) continue;
        if (_is_floor_standing && valid_floor) {
            cv::Vec3d cad_z_cand(R_cand(0, 2), R_cand(1, 2), R_cand(2, 2));
            if (cad_z_cand.dot(n_fl) < 0.70) continue;
        }

        cv::Vec3d t_cand = p_mean - R_cand * q_mean;
        if (!isInTargetWorkspace(t_cand)) continue;

        int inliers = 0;
        double err_sum = 0.0;
        std::vector<uint8_t> mask(N, 0);
        for (size_t i = 0; i < N; ++i) {
            cv::Vec3d pred = R_cand * cv::Vec3d(_cad_centroids[matches[i]]) + t_cand;
            double e = cv::norm(pred - cv::Vec3d(obs_pts[i]));
            if (e <= inlier_thresh) {
                ++inliers;
                err_sum += e * e;
                mask[i] = 1;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_err_sum = err_sum;
            best_mask = std::move(mask);
            best_pose.valid = true;
            best_pose.R = R_cand;
            best_pose.t = t_cand;
            if (best_inliers >= static_cast<int>(0.55 * N)) {
                break;
            }
        }
    }

    // 5. Final Least-Squares Kabsch Refit on ALL RANSAC inliers
    if (best_pose.valid && best_inliers >= 6) {
        cv::Vec3d p_mean(0.0, 0.0, 0.0), q_mean(0.0, 0.0, 0.0);
        int cnt = 0;
        for (size_t i = 0; i < N; ++i) {
            if (!best_mask[i]) continue;
            p_mean += cv::Vec3d(obs_pts[i]);
            q_mean += cv::Vec3d(_cad_centroids[matches[i]]);
            ++cnt;
        }
        if (cnt >= 6) {
            p_mean *= (1.0 / cnt);
            q_mean *= (1.0 / cnt);
            cv::Matx33d H = cv::Matx33d::zeros();
            for (size_t i = 0; i < N; ++i) {
                if (!best_mask[i]) continue;
                cv::Vec3d dp = cv::Vec3d(obs_pts[i]) - p_mean;
                cv::Vec3d dq = cv::Vec3d(_cad_centroids[matches[i]]) - q_mean;
                H += dp * dq.t();
            }
            cv::Matx33d U, Vt;
            cv::Vec3d W;
            cv::SVD::compute(H, W, U, Vt);
            if (W[0] > 1e-6 && W[1] > 1e-7) {
                cv::Matx33d R_refit = U * Vt;
                if (cv::determinant(R_refit) < 0.0) {
                    cv::Matx33d fix = cv::Matx33d::eye();
                    fix(2, 2) = -1.0;
                    R_refit = U * fix * Vt;
                }
                cv::Vec3d t_refit = p_mean - R_refit * q_mean;
                if (isCandidateRotationValid(R_refit) && isInTargetWorkspace(t_refit)) {
                    best_pose.R = R_refit;
                    best_pose.t = t_refit;
                }
            }
        }
    }

    const int min_valid_inliers = std::max(8, static_cast<int>(0.025 * N));
    if (best_pose.valid && best_inliers < min_valid_inliers) {
        best_pose.valid = false;
        return best_pose;
    }

    // 6. Floor upright gravity lock ONLY for FLOOR_STANDING archetypes (tables, racks, pallets)
    if (best_pose.valid && _is_floor_standing && valid_floor) {
        cv::Vec3d z_axis(n_fl[0], n_fl[1], n_fl[2]);
        if (n_fl[1] < -0.82 && n_fl[2] >= 0.05 && n_fl[2] <= 0.35) {
            z_axis = cv::Vec3d(n_fl[0] * 0.5, n_fl[1], n_fl[2]);
        }
        z_axis *= (1.0 / std::max(1e-6, cv::norm(z_axis)));
        cv::Vec3d y_axis(best_pose.R(0, 1), best_pose.R(1, 1), best_pose.R(2, 1));
        y_axis -= z_axis * z_axis.dot(y_axis);
        if (cv::norm(y_axis) > 1e-6) {
            y_axis *= (1.0 / cv::norm(y_axis));
            cv::Vec3d x_axis = y_axis.cross(z_axis);
            best_pose.R = cv::Matx33d(
                x_axis[0], y_axis[0], z_axis[0],
                x_axis[1], y_axis[1], z_axis[1],
                x_axis[2], y_axis[2], z_axis[2]);
        }
    }

    if (best_pose.valid) {
        best_pose.inliers_count = best_inliers;
        best_pose.inlier_ratio = static_cast<double>(best_inliers) / static_cast<double>(std::max<size_t>(1, N));
        best_pose.rmse_m = std::sqrt(best_err_sum / std::max(1, best_inliers));
        best_pose.confidence = std::clamp(static_cast<float>(best_inliers) / static_cast<float>(N), 0.10f, 0.99f);
        best_pose.cluster_points_count = static_cast<int>(obs_pts.size());
        best_pose.t_glass = best_pose.t;
        const bool fold_c2 = (_is_floor_standing && _symmetry_code >= 1);
        cv::Vec3d euler = rotationMatrixToEulerDeg(best_pose.R, fold_c2);
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
    if (!pose.valid || _mesh_pts.empty() || scene_pts.size() < 24) return;

    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 320.0);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 240.0);
    const int img_w = std::max(320, static_cast<int>(2.0 * cx + 0.5));
    const int img_h = std::max(240, static_cast<int>(2.0 * cy + 0.5));

    cv::Matx33d R_curr = pose.R;
    cv::Vec3d t_curr = pose.t;

    // 1. Visibility + Frustum Culling on CAD mesh surface
    std::vector<cv::Vec3d> cad_vis_pts;
    std::vector<cv::Vec3d> cad_vis_norms;
    const size_t mesh_stride = (_mesh_pts.size() > 6000) ? 2 : 1;
    cad_vis_pts.reserve(_mesh_pts.size() / mesh_stride);
    cad_vis_norms.reserve(_mesh_pts.size() / mesh_stride);

    for (size_t i = 0; i < _mesh_pts.size(); i += mesh_stride) {
        cv::Vec3d p_cam = R_curr * cv::Vec3d(_mesh_pts[i]) + t_curr;
        if (p_cam[2] <= 0.05) continue;
        cv::Vec3d n_cam = R_curr * cv::Vec3d(_mesh_normals[i]);
        // Visibility filter: prefer front-facing triangles toward camera optical origin
        if (n_cam.dot(p_cam) > 0.15 * cv::norm(p_cam)) continue;

        double u = p_cam[0] * fx / p_cam[2] + cx;
        double v = p_cam[1] * fy / p_cam[2] + cy;
        if (u >= -40 && u < img_w + 40 && v >= -40 && v < img_h + 40) {
            cad_vis_pts.emplace_back(_mesh_pts[i]);
            cad_vis_norms.emplace_back(_mesh_normals[i]);
        }
    }
    if (cad_vis_pts.size() < 40) return;

    // 2. Scale-adaptive foreground extraction near coarse CAD pose
    const double fg_thresh = std::clamp(_real_bounding_diag_m * 0.032, 0.008, 0.032);
    const double fg_thresh_sq = fg_thresh * fg_thresh;
    cv::Matx33d R_inv = R_curr.t();
    std::vector<cv::Vec3d> fg_pts;
    const size_t scene_step = std::max<size_t>(1, scene_pts.size() / 600);
    fg_pts.reserve(600);

    for (size_t i = 0; i < scene_pts.size(); i += scene_step) {
        cv::Vec3d p_scene(scene_pts[i]);
        cv::Vec3d p_cad = R_inv * (p_scene - t_curr);
        double min_dsq = 1e10;
        for (size_t j = 0; j < cad_vis_pts.size(); j += 4) {
            cv::Vec3d d = p_cad - cad_vis_pts[j];
            double dsq = d.dot(d);
            if (dsq < min_dsq) min_dsq = dsq;
        }
        if (min_dsq <= fg_thresh_sq) {
            fg_pts.push_back(p_scene);
        }
    }
    if (fg_pts.size() < 24) return;

    // 3. Scale-Adaptive Multi-Scale 6DoF Point-to-Plane ICP with Tukey Biweight Loss
    const double scale_k = std::clamp(_real_bounding_diag_m, 0.18, 1.40);
    const double dist_scales[4] = {0.032 * scale_k, 0.016 * scale_k, 0.007 * scale_k, 0.003 * scale_k};

    cv::Matx33d R_icp = R_inv;
    cv::Vec3d t_icp = -R_inv * t_curr;

    for (double max_corr : dist_scales) {
        max_corr = std::max(0.003, max_corr);
        double tukey_k = std::max(0.002, max_corr * 0.65);
        double max_corr_sq = max_corr * max_corr;
        double prev_residual = 1e10;

        for (int iter = 0; iter < 15; ++iter) {
            cv::Matx66d JtJ = cv::Matx66d::zeros();
            cv::Matx61d Jtb = cv::Matx61d::zeros();
            int n_corr = 0;
            double sum_res = 0.0;

            for (const auto& p_sc : fg_pts) {
                cv::Vec3d p_trans = R_icp * p_sc + t_icp;
                double best_dsq = 1e10;
                int best_j = -1;
                for (size_t j = 0; j < cad_vis_pts.size(); j += 2) {
                    cv::Vec3d d = p_trans - cad_vis_pts[j];
                    double dsq = d.dot(d);
                    if (dsq < best_dsq) {
                        best_dsq = dsq;
                        best_j = static_cast<int>(j);
                    }
                }
                if (best_dsq > max_corr_sq || best_j < 0) continue;

                cv::Vec3d n = cad_vis_norms[best_j];
                double n_norm = cv::norm(n);
                if (n_norm < 1e-6) continue;
                n *= (1.0 / n_norm);

                cv::Vec3d res = p_trans - cad_vis_pts[best_j];
                double r = n.dot(res);
                double abs_r = std::abs(r);
                double w = 0.0;
                if (abs_r < tukey_k) {
                    double rk = r / tukey_k;
                    w = (1.0 - rk * rk);
                    w = w * w;
                }
                if (w < 1e-8) continue;

                cv::Vec3d cross_np = p_trans.cross(n);
                double J[6] = {cross_np[0], cross_np[1], cross_np[2], n[0], n[1], n[2]};
                for (int a = 0; a < 6; ++a) {
                    for (int b = 0; b < 6; ++b) {
                        JtJ(a, b) += w * J[a] * J[b];
                    }
                    Jtb(a, 0) -= w * J[a] * r;
                }
                sum_res += w * r * r;
                ++n_corr;
            }

            if (n_corr < 18) break;
            for (int d = 0; d < 6; ++d) JtJ(d, d) += 1e-5;

            cv::Matx61d delta;
            if (!cv::solve(JtJ, Jtb, delta, cv::DECOMP_CHOLESKY)) {
                if (!cv::solve(JtJ, Jtb, delta, cv::DECOMP_SVD)) break;
            }

            cv::Vec3d d_w(delta(0, 0), delta(1, 0), delta(2, 0));
            cv::Vec3d d_t(delta(3, 0), delta(4, 0), delta(5, 0));
            double d_theta = cv::norm(d_w);

            cv::Matx33d d_R = cv::Matx33d::eye();
            if (d_theta > 1e-8) {
                double clamp_th = std::min(d_theta, 0.25);
                cv::Vec3d axis = d_w * (1.0 / d_theta);
                cv::Matx33d K(
                     0.0,     -axis[2],  axis[1],
                     axis[2],  0.0,     -axis[0],
                    -axis[1],  axis[0],  0.0);
                d_R = cv::Matx33d::eye() + std::sin(clamp_th) * K + (1.0 - std::cos(clamp_th)) * (K * K);
            }

            R_icp = d_R * R_icp;
            t_icp = d_R * t_icp + d_t;

            double rmse = std::sqrt(sum_res / n_corr);
            if (cv::norm(d_t) < 1e-5 && d_theta < 1e-5) break;
            if (std::abs(rmse - prev_residual) / (prev_residual + 1e-10) < 1e-5) break;
            prev_residual = rmse;
        }
    }

    cv::Matx33d R_final = R_icp.t();
    cv::Vec3d t_final = -R_final * t_icp;

    const double max_drift = std::clamp(_real_bounding_diag_m * 0.085, 0.025, 0.095);
    if (cv::norm(t_final - pose.t) > max_drift || !isInTargetWorkspace(t_final)) {
        return;
    }

    pose.R = R_final;
    pose.t = t_final;
    pose.t_glass = t_final;
    const bool fold_c2 = (_is_floor_standing && _symmetry_code >= 1);
    cv::Vec3d euler = rotationMatrixToEulerDeg(pose.R, fold_c2);
    pose.roll_deg = euler[0];
    pose.pitch_deg = euler[1];
    pose.yaw_deg = euler[2];
    pose.method = "universal_mesh_icp";
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
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : (0.5 * rgb.cols));
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : (0.5 * rgb.rows));

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
        return (valid > 6) ? (sum / valid) : 1e6;
    };

    const double scale_k = std::clamp(_real_bounding_diag_m, 0.15, 1.15);
    double best_cost = evalChamfer(pose.R, pose.t);
    for (double dz_u : {-0.014, -0.006, 0.0, 0.006, 0.014}) {
        for (double dx_u : {-0.010, 0.0, 0.010}) {
            cv::Vec3d t_try = pose.t + cv::Vec3d(dx_u * scale_k, 0.0, dz_u * scale_k);
            double c = evalChamfer(pose.R, t_try);
            if (c < best_cost) {
                best_cost = c;
                pose.t = t_try;
                pose.t_glass = t_try;
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
    const double fx = (intr.fx > 100.0) ? intr.fx : 608.215;
    const double fy = (intr.fy > 100.0) ? intr.fy : 608.128;
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : 0.5 * img_w);
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : 0.5 * img_h);

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
    const double cx = (intr.ppx > 10.0) ? intr.ppx : ((intr.cx > 10.0) ? intr.cx : (0.5 * img.cols));
    const double cy = (intr.ppy > 10.0) ? intr.ppy : ((intr.cy > 10.0) ? intr.cy : (0.5 * img.rows));

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

    // 3. Scale-adaptive 3D Coordinate Axes at object center
    const double axis_len = std::clamp(_real_bounding_diag_m * 0.16, 0.04, 0.16);
    cv::Point2f p_c = proj_f(0, 0, 0);
    cv::Point2f p_x = proj_f(axis_len, 0, 0);
    cv::Point2f p_y = proj_f(0, axis_len, 0);
    cv::Point2f p_z = proj_f(0, 0, axis_len);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_x), RED, 2, cv::LINE_AA, 0, 0.22);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_y), GREEN, 2, cv::LINE_AA, 0, 0.22);
    cv::arrowedLine(img, to_pt(p_c), to_pt(p_z), BLUE, 2, cv::LINE_AA, 0, 0.22);

    // 4. Universal Object Telemetry Label
    std::ostringstream oss;
    oss << "[" << pose.model_name << "] Z:" << std::fixed << std::setprecision(2) << pose.t[2]
        << "m | Yaw:" << std::setprecision(1) << pose.yaw_deg << " deg";
    cv::Point lbl_pt = (c2d.size() == 8) ? to_pt(c2d[4]) : to_pt(p_c);
    lbl_pt.x = std::clamp(lbl_pt.x, 10, std::max(10, img.cols - 260));
    lbl_pt.y = std::clamp(lbl_pt.y - 10, 24, std::max(24, img.rows - 10));
    cv::putText(img, oss.str(), lbl_pt, cv::FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv::LINE_AA);
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
    const bool fold_c2 = (_is_floor_standing && _symmetry_code >= 1);
    cv::Vec3d euler = rotationMatrixToEulerDeg(pose.R, fold_c2);
    pose.roll_deg = euler[0];
    pose.pitch_deg = euler[1];
    pose.yaw_deg = euler[2];
    pose.bbox_2d = projectCadBoundingBox(pose.R, pose.t, target_intr,
                                         target_width, target_height, &pose.corners_2d);
}

cv::Vec3d UniversalPoseEstimator::rotationMatrixToEulerDeg(const cv::Matx33d& R, bool fold_c2_yaw)
{
    cv::Vec3d v_rail(R(0, 1), R(1, 1), R(2, 1));
    cv::Vec3d v_up(R(0, 2), R(1, 2), R(2, 2));
    double roll = std::asin(std::clamp(v_up[0], -1.0, 1.0)) * 180.0 / kPi;
    double pitch = std::asin(std::clamp(v_up[2], -1.0, 1.0)) * 180.0 / kPi;
    double yaw = std::atan2(v_rail[2], -v_rail[0]) * 180.0 / kPi;
    if (fold_c2_yaw) {
        while (yaw > 90.0) yaw -= 180.0;
        while (yaw < -90.0) yaw += 180.0;
    }
    return cv::Vec3d(roll, pitch, yaw);
}

} // namespace tmcad
