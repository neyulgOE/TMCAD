#include "CADMaskMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

namespace tmcad {

namespace {

cv::Mat letterboxSquare(const cv::Mat& src, int dst_size, const cv::Scalar& bg,
                        int interp, float* out_scale = nullptr,
                        int* out_pad_x = nullptr, int* out_pad_y = nullptr) {
    const int h = src.rows;
    const int w = src.cols;
    const float scale = static_cast<float>(dst_size) / static_cast<float>(std::max(1, std::max(h, w)));
    const int nh = std::max(1, static_cast<int>(std::round(h * scale)));
    const int nw = std::max(1, static_cast<int>(std::round(w * scale)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0.0, 0.0, interp);

    cv::Mat out(dst_size, dst_size, src.type(), bg);
    const int x0 = (dst_size - nw) / 2;
    const int y0 = (dst_size - nh) / 2;
    resized.copyTo(out(cv::Rect(x0, y0, nw, nh)));
    if (out_scale) *out_scale = scale;
    if (out_pad_x) *out_pad_x = x0;
    if (out_pad_y) *out_pad_y = y0;
    return out;
}

cv::Mat letterbox224(const cv::Mat& src, int bg_val, int interp) {
    const int h = src.rows;
    const int w = src.cols;
    const float scale = 200.0f / static_cast<float>(std::max(1, std::max(h, w)));
    const int nh = std::max(1, static_cast<int>(std::round(h * scale)));
    const int nw = std::max(1, static_cast<int>(std::round(w * scale)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0.0, 0.0, interp);

    const cv::Scalar bg = (src.channels() == 1)
        ? cv::Scalar(bg_val)
        : cv::Scalar(bg_val, bg_val, bg_val);
    cv::Mat out(224, 224, src.type(), bg);
    const int x0 = (224 - nw) / 2;
    const int y0 = (224 - nh) / 2;
    resized.copyTo(out(cv::Rect(x0, y0, nw, nh)));
    return out;
}

} // namespace

bool CADMaskMatcher::initBackbones(const std::string& fastsam_onnx_path,
                                   const std::string& dinov2_onnx_path) {
    if (_backbones_ready && _loaded_fastsam_path == fastsam_onnx_path &&
        _loaded_dinov2_path == dinov2_onnx_path) {
        return true;
    }

    try {
        if (!_ort_env) {
            _ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CADMaskMatcher");
        }
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        _fastsam_sess = std::make_unique<Ort::Session>(*_ort_env, fastsam_onnx_path.c_str(), opts);
        _dinov2_sess = std::make_unique<Ort::Session>(*_ort_env, dinov2_onnx_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        _fastsam_in_name = _fastsam_sess->GetInputNameAllocated(0, alloc).get();
        _fastsam_out_names.clear();
        const size_t n_fs_out = _fastsam_sess->GetOutputCount();
        for (size_t i = 0; i < n_fs_out; ++i) {
            _fastsam_out_names.emplace_back(_fastsam_sess->GetOutputNameAllocated(i, alloc).get());
        }
        _dinov2_in_name = _dinov2_sess->GetInputNameAllocated(0, alloc).get();
        _dinov2_out_name = _dinov2_sess->GetOutputNameAllocated(0, alloc).get();

        _loaded_fastsam_path = fastsam_onnx_path;
        _loaded_dinov2_path = dinov2_onnx_path;
        _backbones_ready = (n_fs_out >= 2);
        if (_backbones_ready) {
            std::cout << "[CADMaskMatcher] Loaded shared backbones: FastSAM + DINOv2\n";
        }
        return _backbones_ready;
    } catch (const std::exception& e) {
        std::cerr << "[CADMaskMatcher] Failed to init backbones: " << e.what() << "\n";
        _backbones_ready = false;
        return false;
    }
}

bool CADMaskMatcher::loadTemplates(const std::string& templates_bin_path) {
    _templates_ready = false;
    _templates.clear();

    std::ifstream ifs(templates_bin_path, std::ios::binary);
    if (!ifs.is_open()) return false;

    char magic[8] = {0};
    ifs.read(magic, 8);
    if (!ifs || std::memcmp(magic, "CADMV01\0", 8) != 0) {
        std::cerr << "[CADMaskMatcher] Invalid template magic in " << templates_bin_path << "\n";
        return false;
    }

    uint32_t num_views = 0, feat_dim = 0, mask_size = 0;
    ifs.read(reinterpret_cast<char*>(&num_views), 4);
    ifs.read(reinterpret_cast<char*>(&feat_dim), 4);
    ifs.read(reinterpret_cast<char*>(&mask_size), 4);
    if (!ifs || num_views == 0 || num_views > 2048 || feat_dim == 0 || mask_size == 0) {
        return false;
    }

    _feat_dim = feat_dim;
    _mask_size = mask_size;
    _templates.resize(num_views);
    const size_t m_bytes = static_cast<size_t>(mask_size) * mask_size;

    for (uint32_t i = 0; i < num_views; ++i) {
        float r_buf[9] = {0};
        ifs.read(reinterpret_cast<char*>(r_buf), 9 * sizeof(float));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                _templates[i].R_cw(r, c) = static_cast<double>(r_buf[r * 3 + c]);
            }
        }
        _templates[i].feat.resize(feat_dim);
        ifs.read(reinterpret_cast<char*>(_templates[i].feat.data()), feat_dim * sizeof(float));
        _templates[i].mask_64.resize(m_bytes);
        ifs.read(reinterpret_cast<char*>(_templates[i].mask_64.data()), m_bytes);
        if (!ifs) {
            _templates.clear();
            return false;
        }
    }

    _templates_ready = true;
    std::cout << "[CADMaskMatcher] Loaded " << num_views << " multi-view templates from "
              << templates_bin_path << "\n";
    return true;
}

void CADMaskMatcher::clearTemplates() {
    _templates.clear();
    _templates_ready = false;
}

std::vector<MatchedMaskCandidate> CADMaskMatcher::match(const cv::Mat& rgb_bgr, int top_k) const {
    std::vector<MatchedMaskCandidate> results;
    if (!isReady() || rgb_bgr.empty() || rgb_bgr.channels() != 3) return results;

    const int img_h = rgb_bgr.rows;
    const int img_w = rgb_bgr.cols;
    const double img_area = static_cast<double>(img_h) * img_w;

    // 1. Run FastSAM-s (640x640 letterboxed RGB 0..1)
    float fs_scale = 1.0f;
    int fs_pad_x = 0, fs_pad_y = 0;
    cv::Mat fs_bgr = letterboxSquare(rgb_bgr, 640, cv::Scalar(114, 114, 114),
                                     cv::INTER_LINEAR, &fs_scale, &fs_pad_x, &fs_pad_y);
    cv::Mat fs_rgb;
    cv::cvtColor(fs_bgr, fs_rgb, cv::COLOR_BGR2RGB);

    std::vector<float> fs_input(3 * 640 * 640);
    const int plane = 640 * 640;
    for (int y = 0; y < 640; ++y) {
        const cv::Vec3b* row = fs_rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < 640; ++x) {
            const int idx = y * 640 + x;
            fs_input[idx] = static_cast<float>(row[x][0]) / 255.0f;
            fs_input[plane + idx] = static_cast<float>(row[x][1]) / 255.0f;
            fs_input[2 * plane + idx] = static_cast<float>(row[x][2]) / 255.0f;
        }
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> fs_shape{1, 3, 640, 640};
    Ort::Value fs_tensor = Ort::Value::CreateTensor<float>(
        mem_info, fs_input.data(), fs_input.size(), fs_shape.data(), fs_shape.size());

    const char* fs_in_names[] = {_fastsam_in_name.c_str()};
    std::vector<const char*> fs_out_names;
    for (const auto& s : _fastsam_out_names) fs_out_names.push_back(s.c_str());

    auto fs_outs = _fastsam_sess->Run(
        Ort::RunOptions{nullptr}, fs_in_names, &fs_tensor, 1,
        fs_out_names.data(), fs_out_names.size());
    if (fs_outs.size() < 2) return results;

    const float* det_ptr = fs_outs[0].GetTensorData<float>();   // [1, 37, 8400]
    const float* proto_ptr = fs_outs[1].GetTensorData<float>(); // [1, 32, 160, 160]
    auto det_shape = fs_outs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto proto_shape = fs_outs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (det_shape.size() != 3 || proto_shape.size() != 4) return results;

    const int n_ch = static_cast<int>(det_shape[1]);   // 37 = 4 + 1 + 32
    const int n_anc = static_cast<int>(det_shape[2]);  // 8400
    const int n_proto = static_cast<int>(proto_shape[1]); // 32
    const int ph = static_cast<int>(proto_shape[2]);   // 160
    const int pw = static_cast<int>(proto_shape[3]);   // 160
    if (n_ch < 5 + n_proto) return results;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> anc_indices;
    for (int j = 0; j < n_anc; ++j) {
        const float conf = det_ptr[4 * n_anc + j];
        if (conf < 0.35f) continue;
        const float cx = det_ptr[0 * n_anc + j];
        const float cy = det_ptr[1 * n_anc + j];
        const float bw = det_ptr[2 * n_anc + j];
        const float bh = det_ptr[3 * n_anc + j];
        boxes.emplace_back(
            static_cast<int>(std::round(cx - 0.5f * bw)),
            static_cast<int>(std::round(cy - 0.5f * bh)),
            static_cast<int>(std::round(bw)),
            static_cast<int>(std::round(bh)));
        scores.push_back(conf);
        anc_indices.push_back(j);
    }

    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, scores, 0.35f, 0.70f, nms_idx);
    if (nms_idx.empty()) return results;
    std::sort(nms_idx.begin(), nms_idx.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    if (nms_idx.size() > 30) nms_idx.resize(30);

    struct RawMask {
        cv::Mat mask_u8;
        cv::Rect bbox;
        int area{0};
    };
    std::vector<RawMask> raw_masks;
    raw_masks.reserve(nms_idx.size());

    const float proto_scale = static_cast<float>(pw) / 640.0f;
    for (int k : nms_idx) {
        const int j = anc_indices[k];
        const cv::Rect& b640 = boxes[k];
        const int px0 = std::clamp(static_cast<int>(std::floor(b640.x * proto_scale)), 0, pw - 1);
        const int py0 = std::clamp(static_cast<int>(std::floor(b640.y * proto_scale)), 0, ph - 1);
        const int px1 = std::clamp(static_cast<int>(std::ceil((b640.x + b640.width) * proto_scale)), px0 + 1, pw);
        const int py1 = std::clamp(static_cast<int>(std::ceil((b640.y + b640.height) * proto_scale)), py0 + 1, ph);

        cv::Mat mask160 = cv::Mat::zeros(ph, pw, CV_8UC1);
        for (int py = py0; py < py1; ++py) {
            uint8_t* mrow = mask160.ptr<uint8_t>(py);
            for (int px = px0; px < px1; ++px) {
                float acc = 0.0f;
                const int p_off = py * pw + px;
                for (int c = 0; c < n_proto; ++c) {
                    acc += det_ptr[(5 + c) * n_anc + j] * proto_ptr[c * ph * pw + p_off];
                }
                if (acc > 0.0f) mrow[px] = 255;
            }
        }

        cv::Mat mask640;
        cv::resize(mask160, mask640, cv::Size(640, 640), 0.0, 0.0, cv::INTER_NEAREST);
        const int crop_w = std::max(1, static_cast<int>(std::round(img_w * fs_scale)));
        const int crop_h = std::max(1, static_cast<int>(std::round(img_h * fs_scale)));
        const cv::Rect valid_rect(
            std::clamp(fs_pad_x, 0, 639),
            std::clamp(fs_pad_y, 0, 639),
            std::min(crop_w, 640 - std::clamp(fs_pad_x, 0, 639)),
            std::min(crop_h, 640 - std::clamp(fs_pad_y, 0, 639)));
        cv::Mat mask_full;
        cv::resize(mask640(valid_rect), mask_full, cv::Size(img_w, img_h), 0.0, 0.0, cv::INTER_NEAREST);

        const int area = cv::countNonZero(mask_full);
        if (area < 0.004 * img_area || area > 0.55 * img_area) continue;
        std::vector<cv::Point> nz;
        cv::findNonZero(mask_full, nz);
        if (nz.empty()) continue;
        raw_masks.push_back({std::move(mask_full), cv::boundingRect(nz), area});
    }

    if (raw_masks.empty()) return results;

    // 2. Build single + pixel-adjacent merged candidates
    std::vector<cv::Mat> dilated(raw_masks.size());
    const cv::Mat kern = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    for (size_t i = 0; i < raw_masks.size(); ++i) {
        cv::dilate(raw_masks[i].mask_u8, dilated[i], kern);
    }

    struct CandMask {
        cv::Mat mask_u8;
        cv::Rect bbox;
    };
    std::vector<CandMask> cands;
    cands.reserve(raw_masks.size() * 3);
    for (const auto& rm : raw_masks) {
        cands.push_back({rm.mask_u8, rm.bbox});
    }
    for (size_t i = 0; i < raw_masks.size(); ++i) {
        for (size_t j = i + 1; j < raw_masks.size(); ++j) {
            if (raw_masks[i].area + raw_masks[j].area > 0.45 * img_area) continue;
            // Quick bbox check before pixel touch test
            const cv::Rect expanded(raw_masks[i].bbox.x - 10, raw_masks[i].bbox.y - 10,
                                    raw_masks[i].bbox.width + 20, raw_masks[i].bbox.height + 20);
            if ((expanded & raw_masks[j].bbox).area() <= 0) continue;

            cv::Mat inter;
            cv::bitwise_and(dilated[i], raw_masks[j].mask_u8, inter);
            if (cv::countNonZero(inter) >= 10) {
                cv::Mat merged;
                cv::bitwise_or(raw_masks[i].mask_u8, raw_masks[j].mask_u8, merged);
                cands.push_back({std::move(merged), raw_masks[i].bbox | raw_masks[j].bbox});
            }
        }
    }

    // 3. Batch-encode candidates with DINOv2-vits14 ONNX
    const int n_cand = static_cast<int>(cands.size());
    const int m_sz = static_cast<int>(_mask_size);
    std::vector<float> dino_in(static_cast<size_t>(n_cand) * 3 * 224 * 224);
    std::vector<std::vector<uint8_t>> cand_masks_64(n_cand, std::vector<uint8_t>(m_sz * m_sz, 0));

    const float mean_rgb[3] = {0.485f, 0.456f, 0.406f};
    const float std_rgb[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane224 = 224 * 224;

    for (int idx = 0; idx < n_cand; ++idx) {
        const cv::Rect& b = cands[idx].bbox;
        cv::Mat masked_crop(b.height, b.width, CV_8UC3, cv::Scalar(128, 128, 128));
        rgb_bgr(b).copyTo(masked_crop, cands[idx].mask_u8(b));
        cv::Mat c224 = letterbox224(masked_crop, 128, cv::INTER_AREA);
        cv::Mat c224_rgb;
        cv::cvtColor(c224, c224_rgb, cv::COLOR_BGR2RGB);

        float* dst = dino_in.data() + static_cast<size_t>(idx) * 3 * plane224;
        for (int y = 0; y < 224; ++y) {
            const cv::Vec3b* row = c224_rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < 224; ++x) {
                const int p = y * 224 + x;
                dst[p] = (static_cast<float>(row[x][0]) / 255.0f - mean_rgb[0]) / std_rgb[0];
                dst[plane224 + p] = (static_cast<float>(row[x][1]) / 255.0f - mean_rgb[1]) / std_rgb[1];
                dst[2 * plane224 + p] = (static_cast<float>(row[x][2]) / 255.0f - mean_rgb[2]) / std_rgb[2];
            }
        }

        cv::Mat m224 = letterbox224(cands[idx].mask_u8(b), 0, cv::INTER_NEAREST);
        cv::Mat m64;
        cv::resize(m224, m64, cv::Size(m_sz, m_sz), 0.0, 0.0, cv::INTER_NEAREST);
        for (int y = 0; y < m_sz; ++y) {
            const uint8_t* mrow = m64.ptr<uint8_t>(y);
            for (int x = 0; x < m_sz; ++x) {
                cand_masks_64[idx][y * m_sz + x] = (mrow[x] > 127) ? 1 : 0;
            }
        }
    }

    const std::array<int64_t, 4> dn_shape{n_cand, 3, 224, 224};
    Ort::Value dn_tensor = Ort::Value::CreateTensor<float>(
        mem_info, dino_in.data(), dino_in.size(), dn_shape.data(), dn_shape.size());
    const char* dn_in_names[] = {_dinov2_in_name.c_str()};
    const char* dn_out_names[] = {_dinov2_out_name.c_str()};
    auto dn_outs = _dinov2_sess->Run(
        Ort::RunOptions{nullptr}, dn_in_names, &dn_tensor, 1, dn_out_names, 1);
    const float* emb_ptr = dn_outs[0].GetTensorData<float>();

    // 4. Score against templates: 0.72 * dino_cosine + 0.28 * mask_iou_64
    std::vector<MatchedMaskCandidate> scored;
    scored.reserve(n_cand);
    const int n_tpl = static_cast<int>(_templates.size());
    const int fdim = static_cast<int>(_feat_dim);
    const int m_area = m_sz * m_sz;

    for (int idx = 0; idx < n_cand; ++idx) {
        const float* f = emb_ptr + static_cast<size_t>(idx) * fdim;
        float best_cos = -1.0f;
        int best_t = -1;
        for (int t = 0; t < n_tpl; ++t) {
            const float* tf = _templates[t].feat.data();
            float dot = 0.0f;
            for (int d = 0; d < fdim; ++d) dot += f[d] * tf[d];
            if (dot > best_cos) {
                best_cos = dot;
                best_t = t;
            }
        }
        if (best_t < 0) continue;

        int inter_cnt = 0, union_cnt = 0;
        const auto& cm = cand_masks_64[idx];
        const auto& tm = _templates[best_t].mask_64;
        for (int p = 0; p < m_area; ++p) {
            const int a = cm[p], b = tm[p];
            inter_cnt += (a & b);
            union_cnt += (a | b);
        }
        const float iou = (union_cnt > 0) ? (static_cast<float>(inter_cnt) / union_cnt) : 0.0f;
        const float total_score = 0.72f * best_cos + 0.28f * iou;

        MatchedMaskCandidate mc;
        mc.mask_u8 = cands[idx].mask_u8;
        mc.bbox = cands[idx].bbox;
        mc.R_cw_prior = _templates[best_t].R_cw;
        mc.score = total_score;
        mc.dino_sim = best_cos;
        mc.sil_iou = iou;
        mc.matched_view_idx = best_t;
        scored.push_back(std::move(mc));
    }

    std::sort(scored.begin(), scored.end(), [](const MatchedMaskCandidate& a, const MatchedMaskCandidate& b) {
        return a.score > b.score;
    });

    // Deduplicate overlapping candidates
    for (auto& cand : scored) {
        bool dup = false;
        for (const auto& kept : results) {
            const cv::Rect inter_r = cand.bbox & kept.bbox;
            if (inter_r.area() > 0) {
                const float box_iou = static_cast<float>(inter_r.area()) /
                                      static_cast<float>(cand.bbox.area() + kept.bbox.area() - inter_r.area());
                if (box_iou > 0.75f) {
                    dup = true;
                    break;
                }
            }
        }
        if (!dup) {
            results.push_back(std::move(cand));
            if (static_cast<int>(results.size()) >= top_k) break;
        }
    }
    return results;
}

} // namespace tmcad

