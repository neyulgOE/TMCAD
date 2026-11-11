#include "CADPoseApplication.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <sys/stat.h>

#if !__has_include(<rknn_api.h>) && !__has_include("rknn_api.h")
#define RKNN_SUCC 0
#define RKNN_TENSOR_FLOAT32 0
#define RKNN_TENSOR_NCHW 0
#define RKNN_QUERY_IN_OUT_NUM 0
struct rknn_input {
    uint32_t index;
    void* buf;
    uint32_t size;
    uint8_t pass_through;
    int type;
    int fmt;
};
struct rknn_output {
    uint8_t want_float;
    uint8_t is_prealloc;
    uint32_t index;
    void* buf;
    uint32_t size;
};
struct rknn_input_output_num {
    uint32_t n_input;
    uint32_t n_output;
};
inline int rknn_init(rknn_context*, void*, uint32_t, uint32_t, void*) { return -1; }
inline int rknn_destroy(rknn_context) { return 0; }
inline int rknn_query(rknn_context, int, void*, uint32_t) { return -1; }
inline int rknn_inputs_set(rknn_context, uint32_t, void*) { return -1; }
inline int rknn_run(rknn_context, void*) { return -1; }
inline int rknn_outputs_get(rknn_context, uint32_t, void*, void*) { return -1; }
inline int rknn_outputs_release(rknn_context, uint32_t, void*) { return 0; }
#endif

namespace
{
inline bool fileExists(const std::string& path)
{
    struct stat st{};
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

inline bool dirExists(const std::string& path)
{
    struct stat st{};
    return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

inline std::string joinPath(const std::string& a, const std::string& b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/' || b.front() == '/') return a + b;
    return a + "/" + b;
}
}

CADPoseApplication::~CADPoseApplication()
{
    release();
}

void CADPoseApplication::release()
{
    if (_rknn_loaded && _rknn_ctx != 0)
    {
        rknn_destroy(_rknn_ctx);
        _rknn_ctx = 0;
    }
    _rknn_loaded = false;
    _cad_descriptors.clear();
    _cad_centroids.clear();
    _f2_buf.clear();
    _f3_buf.clear();
}

std::vector<uint8_t> CADPoseApplication::readBinaryFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const auto size = file.tellg();
    if (size <= 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

bool CADPoseApplication::parseNpyCentroids(const std::string& path,
                                           std::vector<cv::Vec3f>& centroids)
{
    centroids.clear();
    std::vector<uint8_t> bytes = readBinaryFile(path);
    if (bytes.size() < 128) return false;

    // Check NPY magic header: \x93NUMPY
    if (bytes[0] != 0x93 || bytes[1] != 'N' || bytes[2] != 'U' ||
        bytes[3] != 'M' || bytes[4] != 'P' || bytes[5] != 'Y')
    {
        return false;
    }

    uint16_t header_len = static_cast<uint16_t>(bytes[8]) |
                          (static_cast<uint16_t>(bytes[9]) << 8);
    size_t data_offset = 10 + header_len;
    if (data_offset >= bytes.size()) return false;

    size_t data_bytes = bytes.size() - data_offset;
    size_t num_floats = data_bytes / sizeof(float);
    if (num_floats % 3 != 0) return false;

    size_t count = num_floats / 3;
    const float* ptr = reinterpret_cast<const float*>(bytes.data() + data_offset);

    centroids.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        centroids.emplace_back(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
    }
    return true;
}

bool CADPoseApplication::initialize(const DetectorConfig& config,
                                    std::string& error)
{
    release();
    _config = config;
    error.clear();

    // 1. Locate package directory
    std::vector<std::string> candidate_roots = {
        "/home/orangepi/Desktop/TMVisionary/models",
        "./models",
        "../models",
        "/home/huyen1905/TMCAD/ObjectDetection_V2.1/models",
        "/home/huyen1905/TMCAD/ObjectDetection_V2.0/models",
        "/home/huyen1905/TMCAD/ObjectDetection_V1.0/models",
        "/home/huyen1905/TMCAD/deploy_orange_pi_5b"
    };

    std::string pkg_dir;
    for (const auto& root : candidate_roots)
    {
        std::string test_dir = joinPath(root, config.model_package);
        if (dirExists(test_dir))
        {
            pkg_dir = test_dir;
            break;
        }
    }

    if (pkg_dir.empty())
    {
        // Check if config.model_package itself is an absolute directory
        if (dirExists(config.model_package))
        {
            pkg_dir = config.model_package;
        }
        else
        {
            error = "cannot find CAD model package directory: " + config.model_package;
            return false;
        }
    }

    // 2. Load RKNN Model for NPU
    std::string rknn_path = joinPath(pkg_dir, "artifacts/orangepi5b_rk3588/model_obs_rk3588.rknn");
    if (!fileExists(rknn_path))
    {
        // Fallback for direct deploy folder
        rknn_path = joinPath(pkg_dir, "model_obs_rk3588.rknn");
    }

    if (!fileExists(rknn_path))
    {
        error = "RKNN model file not found: " + rknn_path;
        return false;
    }

    std::vector<uint8_t> model_data = readBinaryFile(rknn_path);
    if (model_data.empty())
    {
        error = "failed to read RKNN model data from " + rknn_path;
        return false;
    }

    int ret = rknn_init(&_rknn_ctx,
                        model_data.data(),
                        static_cast<uint32_t>(model_data.size()),
                        0,
                        nullptr);
    if (ret != RKNN_SUCC)
    {
        std::ostringstream ss;
        ss << "rknn_init failed with code " << ret;
        error = ss.str();
        _rknn_ctx = 0;
        return false;
    }
    _rknn_loaded = true;

    rknn_input_output_num io_num{};
    if (rknn_query(_rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) == RKNN_SUCC)
    {
        _n_input = io_num.n_input;
    }
    else
    {
        _n_input = 3;
    }

    // 3. Load Precomputed CAD Descriptors (512 * 128 floats = 262144 bytes)
    std::string desc_path = joinPath(pkg_dir, "cad_data/TABLE-DEMO_cad_descriptors.bin");
    if (!fileExists(desc_path))
    {
        desc_path = joinPath(pkg_dir, "TABLE-DEMO_cad_descriptors.bin");
    }

    if (!fileExists(desc_path))
    {
        error = "CAD descriptors file not found: " + desc_path;
        release();
        return false;
    }

    std::vector<uint8_t> desc_bytes = readBinaryFile(desc_path);
    size_t expected_bytes = static_cast<size_t>(_num_cad_samples * _desc_dim) * sizeof(float);
    if (desc_bytes.size() < expected_bytes)
    {
        error = "CAD descriptors file size mismatch: expected " + std::to_string(expected_bytes) +
                " bytes, got " + std::to_string(desc_bytes.size());
        release();
        return false;
    }

    _cad_descriptors.resize(_num_cad_samples * _desc_dim);
    std::memcpy(_cad_descriptors.data(), desc_bytes.data(), expected_bytes);

    // 4. Load CAD Key Centroids (512 * 3)
    std::string centroids_path = joinPath(pkg_dir, "cad_data/TABLE-DEMO_key_centroids.npy");
    if (!fileExists(centroids_path))
    {
        centroids_path = joinPath(pkg_dir, "TABLE-DEMO_key_centroids.npy");
    }

    if (fileExists(centroids_path))
    {
        parseNpyCentroids(centroids_path, _cad_centroids);
    }

    if (_cad_centroids.size() != static_cast<size_t>(_num_cad_samples))
    {
        // Fallback default distribution along table surface if file missing
        _cad_centroids.resize(_num_cad_samples);
        for (int i = 0; i < _num_cad_samples; ++i)
        {
            float fx = static_cast<float>((i % 16) - 8) * 0.025f;
            float fy = static_cast<float>((i / 16) - 16) * 0.025f;
            _cad_centroids[i] = cv::Vec3f(fx, fy, static_cast<float>(_cad_top_z_m));
        }
    }

    if (config.conf_threshold > 0.0f)
    {
        _min_inlier_ratio = static_cast<double>(config.conf_threshold);
    }

    // 5. Load table dimensions from table_dimensions.json if available
    std::string dims_path = joinPath(pkg_dir, "cad_data/table_dimensions.json");
    if (!fileExists(dims_path))
    {
        dims_path = joinPath(pkg_dir, "table_dimensions.json");
    }
    if (fileExists(dims_path))
    {
        cv::FileStorage fs(dims_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (fs.isOpened())
        {
            if (!fs["width_m"].empty()) _cad_half_width_m = static_cast<double>(fs["width_m"]) * 0.5;
            if (!fs["length_m"].empty()) _cad_half_length_m = static_cast<double>(fs["length_m"]) * 0.5;
            if (!fs["tabletop_center_offset_z_m"].empty()) _cad_top_z_m = static_cast<double>(fs["tabletop_center_offset_z_m"]);
            if (!fs["ground_contact_offset_z_m"].empty()) _cad_bottom_z_m = static_cast<double>(fs["ground_contact_offset_z_m"]);
            if (!fs["key_points_count"].empty()) _num_cad_samples = static_cast<int>(fs["key_points_count"]);
            if (!fs["descriptor_dim"].empty()) _desc_dim = static_cast<int>(fs["descriptor_dim"]);
        }
    }

    // 6. Preallocate multi-input buffers for zero-allocation inference
    if (_n_input >= 3)
    {
        static const size_t f2_size = 1 * 128 * 1024 * 20 * sizeof(float);
        static const size_t f3_size = 1 * 256 * 1024 * 20 * sizeof(float);
        _f2_buf.assign(f2_size, 0);
        _f3_buf.assign(f3_size, 0);
    }

    return true;
}

std::vector<cv::Vec3f> CADPoseApplication::extractObservationPoints(const RgbdFrame& frame) const
{
    std::vector<cv::Vec3f> all_pts;
    const int w = static_cast<int>(frame.width);
    const int h = static_cast<int>(frame.height);
    const double fx = frame.intrinsics.fx;
    const double fy = frame.intrinsics.fy;
    const double cx = frame.intrinsics.ppx;
    const double cy = frame.intrinsics.ppy;
    const double scale = frame.depth_scale;

    all_pts.reserve(10000);

    // Step 4 pixels for efficient spatial coverage
    const int step = 4;
    for (int y = 0; y < h; y += step)
    {
        const uint16_t* depth_row = frame.depth_z16.ptr<uint16_t>(y);
        for (int x = 0; x < w; x += step)
        {
            uint16_t d_raw = depth_row[x];
            if (d_raw == 0) continue;

            float z = static_cast<float>(d_raw * scale);
            if (z < 0.35f || z > 2.50f) continue; // Working depth range

            float px = static_cast<float>((x - cx) * z / fx);
            float py = static_cast<float>((y - cy) * z / fy);
            all_pts.emplace_back(px, py, z);
        }
    }

    if (all_pts.size() < static_cast<size_t>(_num_obs_points))
    {
        return all_pts;
    }

    // Uniformly subsample down to exactly _num_obs_points (1024)
    std::vector<cv::Vec3f> sampled;
    sampled.reserve(_num_obs_points);
    double stride = static_cast<double>(all_pts.size()) / static_cast<double>(_num_obs_points);
    for (int i = 0; i < _num_obs_points; ++i)
    {
        size_t idx = static_cast<size_t>(i * stride);
        if (idx >= all_pts.size()) idx = all_pts.size() - 1;
        sampled.push_back(all_pts[idx]);
    }
    return sampled;
}

std::vector<float> CADPoseApplication::buildGraphFeatures(const std::vector<cv::Vec3f>& pts,
                                                           int k) const
{
    const int N = static_cast<int>(pts.size());
    // Output NCHW: (1, 7, N, k) = 7 * N * k floats
    std::vector<float> tensor_nchw(7 * N * k, 0.0f);

    std::vector<std::pair<float, int>> dists(N);

    for (int i = 0; i < N; ++i)
    {
        const cv::Vec3f& pi = pts[i];
        for (int j = 0; j < N; ++j)
        {
            cv::Vec3f diff = pts[j] - pi;
            float d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
            dists[j] = {d2, j};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        for (int ki = 0; ki < k; ++ki)
        {
            int nb_idx = dists[ki].second;
            const cv::Vec3f& pj = pts[nb_idx];
            cv::Vec3f diff = pj - pi;
            float dist = std::sqrt(dists[ki].first);

            // Channel 0..2: x_i (3)
            tensor_nchw[(0 * N + i) * k + ki] = pi[0];
            tensor_nchw[(1 * N + i) * k + ki] = pi[1];
            tensor_nchw[(2 * N + i) * k + ki] = pi[2];

            // Channel 3..5: diff = x_j - x_i (3)
            tensor_nchw[(3 * N + i) * k + ki] = diff[0];
            tensor_nchw[(4 * N + i) * k + ki] = diff[1];
            tensor_nchw[(5 * N + i) * k + ki] = diff[2];

            // Channel 6: distance (1)
            tensor_nchw[(6 * N + i) * k + ki] = dist;
        }
    }
    return tensor_nchw;
}

bool CADPoseApplication::runNpuInference(const std::vector<float>& graph_tensor,
                                         std::vector<float>& out_descriptors,
                                         std::string& error)
{
    if (!_rknn_loaded || _rknn_ctx == 0)
    {
        error = "RKNN context is not initialized";
        return false;
    }

    int ret = RKNN_SUCC;
    if (_n_input >= 3)
    {
        std::vector<rknn_input> inputs(3);
        std::memset(inputs.data(), 0, sizeof(rknn_input) * 3);

        // Input 0: f1 (1, 7, 1024, 20)
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].fmt = RKNN_TENSOR_NCHW;
        inputs[0].size = static_cast<uint32_t>(graph_tensor.size() * sizeof(float));
        inputs[0].buf = const_cast<float*>(graph_tensor.data());
        inputs[0].pass_through = 0;

        // Input 1: f2 (1, 128, 1024, 20)
        inputs[1].index = 1;
        inputs[1].type = RKNN_TENSOR_FLOAT32;
        inputs[1].fmt = RKNN_TENSOR_NCHW;
        inputs[1].size = static_cast<uint32_t>(_f2_buf.size());
        inputs[1].buf = _f2_buf.data();
        inputs[1].pass_through = 0;

        // Input 2: f3 (1, 256, 1024, 20)
        inputs[2].index = 2;
        inputs[2].type = RKNN_TENSOR_FLOAT32;
        inputs[2].fmt = RKNN_TENSOR_NCHW;
        inputs[2].size = static_cast<uint32_t>(_f3_buf.size());
        inputs[2].buf = _f3_buf.data();
        inputs[2].pass_through = 0;

        ret = rknn_inputs_set(_rknn_ctx, 3, inputs.data());
        if (ret != RKNN_SUCC)
        {
            error = "rknn_inputs_set failed: code " + std::to_string(ret);
            return false;
        }
    }
    else
    {
        rknn_input input;
        std::memset(&input, 0, sizeof(input));
        input.index = 0;
        input.type = RKNN_TENSOR_FLOAT32;
        input.fmt = RKNN_TENSOR_NCHW;
        input.size = static_cast<uint32_t>(graph_tensor.size() * sizeof(float));
        input.buf = const_cast<float*>(graph_tensor.data());
        input.pass_through = 0;

        ret = rknn_inputs_set(_rknn_ctx, 1, &input);
        if (ret != RKNN_SUCC)
        {
            error = "rknn_inputs_set failed: code " + std::to_string(ret);
            return false;
        }
    }

    ret = rknn_run(_rknn_ctx, nullptr);
    if (ret != RKNN_SUCC)
    {
        error = "rknn_run failed: code " + std::to_string(ret);
        return false;
    }

    rknn_output output;
    std::memset(&output, 0, sizeof(output));
    output.index = 0;
    output.want_float = 1;
    output.is_prealloc = 0;

    ret = rknn_outputs_get(_rknn_ctx, 1, &output, nullptr);
    if (ret != RKNN_SUCC)
    {
        error = "rknn_outputs_get failed: code " + std::to_string(ret);
        return false;
    }

    const size_t elem_count = static_cast<size_t>(_num_obs_points * _desc_dim);
    const float* val_ptr = static_cast<const float*>(output.buf);

    if (!val_ptr)
    {
        rknn_outputs_release(_rknn_ctx, 1, &output);
        error = "RKNN output buffer is null";
        return false;
    }

    out_descriptors.assign(val_ptr, val_ptr + elem_count);
    rknn_outputs_release(_rknn_ctx, 1, &output);
    return true;
}

CADPoseApplication::Pose6D
CADPoseApplication::solveRansacKabsch(const std::vector<cv::Vec3f>& obs_points,
                                      const std::vector<float>& obs_descriptors) const
{
    Pose6D best_pose;
    const int N_obs = static_cast<int>(obs_points.size());
    const int N_cad = _num_cad_samples;
    const int D = _desc_dim;

    if (N_obs < 16 || _cad_centroids.size() != static_cast<size_t>(N_cad))
    {
        return best_pose;
    }

    // 1. Compute cosine similarity matrix S = D_obs * D_cad^T (N_obs x N_cad)
    // and find top-1 CAD correspondence for each observation point
    std::vector<int> top_matches(N_obs);
    std::vector<float> top_scores(N_obs);

    for (int i = 0; i < N_obs; ++i)
    {
        const float* d_obs = obs_descriptors.data() + (i * D);
        float best_sim = -1e9f;
        int best_j = 0;

        for (int j = 0; j < N_cad; ++j)
        {
            const float* d_cad = _cad_descriptors.data() + (j * D);
            float dot = 0.0f;
            for (int k = 0; k < D; ++k)
            {
                dot += d_obs[k] * d_cad[k];
            }
            if (dot > best_sim)
            {
                best_sim = dot;
                best_j = j;
            }
        }
        top_matches[i] = best_j;
        top_scores[i] = best_sim;
    }

    // 2. RANSAC Kabsch SVD
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_idx(0, N_obs - 1);

    int best_inliers_count = 0;
    std::vector<uint8_t> best_inlier_mask(N_obs, 0);

    for (int iter = 0; iter < _ransac_max_iterations; ++iter)
    {
        // Randomly pick 4 distinct correspondence pairs
        int sample_indices[4];
        for (int s = 0; s < 4; ++s)
        {
            sample_indices[s] = dist_idx(rng);
        }

        // Centers
        cv::Vec3d mu_obs{0.0, 0.0, 0.0};
        cv::Vec3d mu_cad{0.0, 0.0, 0.0};
        for (int s = 0; s < 4; ++s)
        {
            int idx = sample_indices[s];
            mu_obs += cv::Vec3d(obs_points[idx]);
            mu_cad += cv::Vec3d(_cad_centroids[top_matches[idx]]);
        }
        mu_obs *= 0.25;
        mu_cad *= 0.25;

        // Covariance matrix H
        cv::Matx33d H = cv::Matx33d::zeros();
        for (int s = 0; s < 4; ++s)
        {
            int idx = sample_indices[s];
            cv::Vec3d p_obs = cv::Vec3d(obs_points[idx]) - mu_obs;
            cv::Vec3d p_cad = cv::Vec3d(_cad_centroids[top_matches[idx]]) - mu_cad;
            H += p_obs * p_cad.t();
        }

        // SVD of H: H = U * W * V^T -> R = U * V^T
        cv::Matx33d U, Vt;
        cv::Vec3d W;
        cv::SVD::compute(H, W, U, Vt);
        cv::Matx33d R_candidate = U * Vt;

        if (cv::determinant(R_candidate) < 0.0)
        {
            cv::Matx33d fix = cv::Matx33d::eye();
            fix(2, 2) = -1.0;
            R_candidate = U * fix * Vt;
        }

        cv::Vec3d t_candidate = mu_obs - R_candidate * mu_cad;

        // Count inliers
        int current_inliers = 0;
        std::vector<uint8_t> current_mask(N_obs, 0);

        for (int i = 0; i < N_obs; ++i)
        {
            cv::Vec3d expected = R_candidate * cv::Vec3d(_cad_centroids[top_matches[i]]) + t_candidate;
            cv::Vec3d diff = cv::Vec3d(obs_points[i]) - expected;
            double err = cv::norm(diff);
            if (err < _ransac_dist_threshold_m)
            {
                ++current_inliers;
                current_mask[i] = 1;
            }
        }

        if (current_inliers > best_inliers_count)
        {
            best_inliers_count = current_inliers;
            best_inlier_mask = std::move(current_mask);
            best_pose.R = R_candidate;
            best_pose.t = t_candidate;
        }
    }

    double ratio = static_cast<double>(best_inliers_count) / static_cast<double>(N_obs);
    best_pose.inlier_ratio = ratio;
    best_pose.inliers_count = best_inliers_count;

    if (ratio < _min_inlier_ratio || best_inliers_count < 16)
    {
        best_pose.valid = false;
        return best_pose;
    }

    // 3. Final refinement: Kabsch SVD on all consensus inliers
    cv::Vec3d mu_obs{0.0, 0.0, 0.0};
    cv::Vec3d mu_cad{0.0, 0.0, 0.0};
    for (int i = 0; i < N_obs; ++i)
    {
        if (best_inlier_mask[i])
        {
            mu_obs += cv::Vec3d(obs_points[i]);
            mu_cad += cv::Vec3d(_cad_centroids[top_matches[i]]);
        }
    }
    mu_obs *= (1.0 / best_inliers_count);
    mu_cad *= (1.0 / best_inliers_count);

    cv::Matx33d H = cv::Matx33d::zeros();
    for (int i = 0; i < N_obs; ++i)
    {
        if (best_inlier_mask[i])
        {
            cv::Vec3d p_obs = cv::Vec3d(obs_points[i]) - mu_obs;
            cv::Vec3d p_cad = cv::Vec3d(_cad_centroids[top_matches[i]]) - mu_cad;
            H += p_obs * p_cad.t();
        }
    }

    cv::Matx33d U, Vt;
    cv::Vec3d W;
    cv::SVD::compute(H, W, U, Vt);
    cv::Matx33d R_final = U * Vt;
    if (cv::determinant(R_final) < 0.0)
    {
        cv::Matx33d fix = cv::Matx33d::eye();
        fix(2, 2) = -1.0;
        R_final = U * fix * Vt;
    }

    best_pose.R = R_final;
    best_pose.t = mu_obs - R_final * mu_cad;

    // Calculate inlier RMSE
    double sum_err_sq = 0.0;
    for (int i = 0; i < N_obs; ++i)
    {
        if (best_inlier_mask[i])
        {
            cv::Vec3d expected = best_pose.R * cv::Vec3d(_cad_centroids[top_matches[i]]) + best_pose.t;
            cv::Vec3d diff = cv::Vec3d(obs_points[i]) - expected;
            sum_err_sq += (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        }
    }
    best_pose.rmse_m = std::sqrt(sum_err_sq / best_inliers_count);

    // Extract Euler angles in degrees
    cv::Vec3d euler = rotationMatrixToEulerDeg(best_pose.R);
    best_pose.roll_deg = euler[0];
    best_pose.pitch_deg = euler[1];
    best_pose.yaw_deg = euler[2];
    best_pose.valid = true;

    return best_pose;
}

cv::Vec3d CADPoseApplication::rotationMatrixToEulerDeg(const cv::Matx33d& R)
{
    double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6;

    double roll, pitch, yaw;
    if (!singular)
    {
        roll = std::atan2(R(2, 1), R(2, 2)) * 180.0 / CV_PI;
        pitch = std::atan2(-R(2, 0), sy) * 180.0 / CV_PI;
        yaw = std::atan2(R(1, 0), R(0, 0)) * 180.0 / CV_PI;
    }
    else
    {
        roll = std::atan2(-R(1, 2), R(1, 1)) * 180.0 / CV_PI;
        pitch = std::atan2(-R(2, 0), sy) * 180.0 / CV_PI;
        yaw = 0.0;
    }
    return cv::Vec3d(roll, pitch, yaw);
}

cv::Rect2f CADPoseApplication::projectCadBoundingBox(const cv::Matx33d& R,
                                                     const cv::Vec3d& t,
                                                     const CameraIntrinsics& intr,
                                                     int img_w, int img_h) const
{
    // 8 3D corners of the CAD bounding box
    const double hx = _cad_half_width_m;
    const double hy = _cad_half_length_m;
    const double z_top = _cad_top_z_m;
    const double z_bot = _cad_bottom_z_m;

    std::vector<cv::Vec3d> corners = {
        {-hx, -hy, z_bot}, {hx, -hy, z_bot}, {hx, hy, z_bot}, {-hx, hy, z_bot},
        {-hx, -hy, z_top}, {hx, -hy, z_top}, {hx, hy, z_top}, {-hx, hy, z_top}
    };

    float min_u = static_cast<float>(img_w);
    float max_u = 0.0f;
    float min_v = static_cast<float>(img_h);
    float max_v = 0.0f;

    for (const auto& c : corners)
    {
        cv::Vec3d pt_cam = R * c + t;
        if (pt_cam[2] <= 0.05) continue;

        float u = static_cast<float>(intr.fx * pt_cam[0] / pt_cam[2] + intr.ppx);
        float v = static_cast<float>(intr.fy * pt_cam[1] / pt_cam[2] + intr.ppy);

        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    min_u = std::clamp(min_u, 0.0f, static_cast<float>(img_w - 1));
    max_u = std::clamp(max_u, 0.0f, static_cast<float>(img_w - 1));
    min_v = std::clamp(min_v, 0.0f, static_cast<float>(img_h - 1));
    max_v = std::clamp(max_v, 0.0f, static_cast<float>(img_h - 1));

    if (max_u <= min_u || max_v <= min_v)
    {
        return cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f);
    }

    return cv::Rect2f(min_u, min_v, max_u - min_u, max_v - min_v);
}

bool CADPoseApplication::process(const RgbdFrame& frame,
                                 DetectorOutput& output,
                                 std::string& error)
{
    error.clear();
    output.application_requested = true;
    output.application_applied = false;

    // 1. Preprocessing: extract 3D observation point cloud from depth_z16
    std::vector<cv::Vec3f> obs_points = extractObservationPoints(frame);
    if (obs_points.size() < static_cast<size_t>(_num_obs_points))
    {
        output.application_status = "insufficient_depth_points";
        return true;
    }

    // 2. Build graph features (1, 7, 1024, 20)
    std::vector<float> graph_tensor = buildGraphFeatures(obs_points, _obs_k);

    // 3. Run RKNN NPU inference (~3.5 ms on RK3588)
    std::vector<float> obs_desc;
    if (!runNpuInference(graph_tensor, obs_desc, error))
    {
        output.application_status = "npu_inference_failed";
        return false;
    }

    // 4. Solve 6DoF Pose via RANSAC Kabsch SVD
    Pose6D pose = solveRansacKabsch(obs_points, obs_desc);
    if (!pose.valid)
    {
        output.application_status = "no_cad_object_detected";
        return true;
    }

    // 5. Apply mechanical camera offset (Intel RealSense D435 Figure 4-6)
    // Offset +17.5 mm from Left IR optical axis to Centerline 1/4-20 front glass center
    cv::Vec3d t_glass;
    t_glass[0] = pose.t[0] + _cam_offset_x_m;
    t_glass[1] = pose.t[1] + _cam_offset_y_m;
    t_glass[2] = pose.t[2] + _cam_offset_z_m;

    // 6. Reproject 3D CAD Bounding Box to 2D image coordinates on RGB
    int img_w = static_cast<int>(frame.width);
    int img_h = static_cast<int>(frame.height);
    cv::Rect2f bbox_2d = projectCadBoundingBox(pose.R, t_glass, frame.intrinsics, img_w, img_h);

    // 7. Populate full 6DoF DetectionObject
    DetectionObject cad_obj;
    cad_obj.class_id = 0;
    cad_obj.label = "table";
    cad_obj.confidence = static_cast<float>(pose.inlier_ratio);

    // Full 3D Translation from Camera Front Glass Center (m)
    cad_obj.has_3d = true;
    cad_obj.x_3d = static_cast<float>(t_glass[0]);
    cad_obj.y_3d = static_cast<float>(t_glass[1]);
    cad_obj.z_3d = static_cast<float>(t_glass[2]);
    cad_obj.depth_m = cad_obj.z_3d;

    // Full 3D Rotation Euler Angles (deg)
    cad_obj.has_pose_6d = true;
    cad_obj.roll_deg = static_cast<float>(pose.roll_deg);
    cad_obj.pitch_deg = static_cast<float>(pose.pitch_deg);
    cad_obj.yaw_deg = static_cast<float>(pose.yaw_deg);
    cad_obj.pose_frame_id = "camera_front_glass_center";

    // 2D Bounding Box (reprojected directly from 3D CAD)
    cad_obj.x = bbox_2d.x;
    cad_obj.y = bbox_2d.y;
    cad_obj.width = bbox_2d.width;
    cad_obj.height = bbox_2d.height;
    cad_obj.center_x = bbox_2d.x + bbox_2d.width * 0.5f;
    cad_obj.center_y = bbox_2d.y + bbox_2d.height * 0.5f;

    output.result.objects.clear();
    output.result.objects.push_back(cad_obj);
    output.pose_object_count = 1;
    output.xyz_object_count = 1;
    output.application_applied = true;
    output.application_status = "ok";

    return true;
}
