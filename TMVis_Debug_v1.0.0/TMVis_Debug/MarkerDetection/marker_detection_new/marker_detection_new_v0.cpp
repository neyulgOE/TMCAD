#include "MarkerDetection/marker_detection_new.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <random> // Thêm thư viện random cho RANSAC

// ====================== ROI CỐ ĐỊNH CHO 3D SAFETY ======================
//const cv::Rect SAFETY_ROI(195, 115, 250, 250); 
// =======================================================================

ArucoDetectorNew::ArucoDetectorNew(const std::string& name)
    : Plugin(name)
{
    loadParams();

    _dict = cv::aruco::getPredefinedDictionary(
        static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(_dictId));

    _params = cv::aruco::DetectorParameters::create();
    _params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
}

ArucoDetectorNew::~ArucoDetectorNew()
{
    _running.store(false);
}

void ArucoDetectorNew::loadParams()
{
    _dictId = static_cast<int>(cv::aruco::DICT_6X6_100);
    _loopSleepMs = 100;
    _maxSyncDiffMs = 50;
    _roiHalfWin = 3;
    _enableXYZ = true;
    _enableStdout = true;

    _maxSyncDiffNs = static_cast<uint64_t>(_maxSyncDiffMs) * 1000000ULL;
}

void ArucoDetectorNew::run()
{
    _running.store(true);
    if (_enableStdout) std::cout << "[ArucoNew][Running] - MASK + RANSAC + OLS (3 MARKERS MODE) Activated\n";

    while (_running.load()) {
        tmv::DepthFrame d;
        tmv::ImageFrame r;
        bool okD = getSubscriberData(d);
        bool okR = getSubscriberRGBD(r);
        if (okD && okR) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    while (_running.load()) {
        tmv::MarkerDetections det;
        // Logic in Log đã được gộp siêu đẹp vào trong tryBuildDetection,
        // nên ở đây ta gọi hàm và bỏ qua việc in lặp lại để Terminal sạch sẽ!
        tryBuildDetection(det); 
        
        std::this_thread::sleep_for(std::chrono::milliseconds(_loopSleepMs));
    }
}

bool ArucoDetectorNew::tryBuildDetection(tmv::MarkerDetections& out)
{
    tmv::DepthFrame dmsg;
    tmv::ImageFrame rmsg;

    if (!getSubscriberData(dmsg)) return false;
    if (!getSubscriberRGBD(rmsg)) return false;

    if (!dmsg.valid || !rmsg.valid) return false;
    if (dmsg.depth_z16.empty() || rmsg.bgr8.empty()) return false;

    const uint64_t dns = dmsg.header.data_nsec;
    const uint64_t rns = rmsg.header.data_nsec;
    const uint64_t diff = (dns > rns) ? (dns - rns) : (rns - dns);
    if (diff > _maxSyncDiffNs) return false;

    const uint64_t procKey = dns ^ (rns << 1);
    if (procKey != 0 && procKey == _lastProcKey) return false;
    _lastProcKey = procKey;

    const int w = static_cast<int>(rmsg.width);
    const int h = static_cast<int>(rmsg.height);
    const int stride = static_cast<int>(rmsg.stride_bytes > 0 ? rmsg.stride_bytes : w * 3);

    if (w <= 0 || h <= 0) return false;
    if (static_cast<int>(rmsg.bgr8.size()) < h * stride) return false;

    cv::Mat bgrView(h, w, CV_8UC3, (void*)rmsg.bgr8.data(), stride);
    cv::Mat bgr = bgrView.clone();

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // =======================================================
    // 1. NHẬN DIỆN CẢ 2 TỪ ĐIỂN SONG SONG
    // =======================================================
    auto dict_5x5 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    auto dict_6x6 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);

    std::vector<int> ids_5, ids_6;
    std::vector<std::vector<cv::Point2f>> corners_5, corners_6;
    cv::aruco::detectMarkers(gray, dict_5x5, corners_5, ids_5, _params);
    cv::aruco::detectMarkers(gray, dict_6x6, corners_6, ids_6, _params);

    out = tmv::MarkerDetections{};
    out.header.pub_nsec = dns;
    out.header.data_nsec = dns;
    out.width = static_cast<uint32_t>(w);
    out.height = static_cast<uint32_t>(h);
    out.valid = true;

    if (ids_5.empty() && ids_6.empty()) return true;

    const float fx = dmsg.intr.fx;
    const float fy = dmsg.intr.fy;
    const float ppx = dmsg.intr.ppx;
    const float ppy = dmsg.intr.ppy;

    // --- BIẾN LƯU TRỮ ĐỂ TÍNH SO SÁNH ---
    int id_center = 3, id_left = 2, id_right = 1;
    float center_3d[3] = {0}, L_3d[3] = {0}, R_3d[3] = {0};
    bool has_center = false, has_L = false, has_R = false;

    // =======================================================
    // 2. HÀM LAMBDA XỬ LÝ CHUNG CHO CÁC MARKER
    // =======================================================
    auto process_markers = [&](const std::vector<int>& ids, const std::vector<std::vector<cv::Point2f>>& corners, bool is_5x5) {
        for (size_t i = 0; i < ids.size(); ++i) {
            if (corners[i].size() != 4) continue;

            cv::Point2f c(0.0f, 0.0f);
            for (int k = 0; k < 4; ++k) c += corners[i][k];
            c *= 0.25f;

            tmv::Marker mk;
            mk.id = ids[i];

            int cu = clampi(static_cast<int>(std::lround(c.x)), 0, w - 1);
            int cvv = clampi(static_cast<int>(std::lround(c.y)), 0, h - 1);

            mk.center.u = static_cast<uint32_t>(cu);
            mk.center.v = static_cast<uint32_t>(cvv);

            auto fillUV = [&](tmv::PixelUV& p, const cv::Point2f& pt) {
                p.u = static_cast<uint32_t>(clampi(static_cast<int>(std::lround(pt.x)), 0, w - 1));
                p.v = static_cast<uint32_t>(clampi(static_cast<int>(std::lround(pt.y)), 0, h - 1));
            };

            fillUV(mk.c0, corners[i][0]);
            fillUV(mk.c1, corners[i][1]);
            fillUV(mk.c2, corners[i][2]);
            fillUV(mk.c3, corners[i][3]);

            if (_enableXYZ) {
                float refined_z = getPlaneDepthRobust(dmsg, corners[i]);
                float X = 0.0f, Y = 0.0f, Z = 0.0f;
                bool ok = false;
                
                if (refined_z > 0.0f) {
                    ok = deprojectXYZ(X, Y, Z, static_cast<float>(cu), static_cast<float>(cvv), refined_z, fx, fy, ppx, ppy);
                }

                mk.has_xyz = ok;
                if (ok) {
                    // Trừ bù cơ khí (Đổi 0.0042 thành 0.005 cho chuẩn với bài test)
                    X -= 0.0175f;
                    Y -= 0.0f;
                    Z -= 0.005f; 

                    mk.x = X; mk.y = Y; mk.z = Z;

                    // Phân loại để lưu biến
                    if (is_5x5 && mk.id == id_center) {
                        center_3d[0] = X; center_3d[1] = Y; center_3d[2] = Z; has_center = true;
                    } else if (!is_5x5 && mk.id == id_left) {
                        L_3d[0] = X; L_3d[1] = Y; L_3d[2] = Z; has_L = true;
                    } else if (!is_5x5 && mk.id == id_right) {
                        R_3d[0] = X; R_3d[1] = Y; R_3d[2] = Z; has_R = true;
                    }
                }
            }
            out.markers.push_back(mk); // Add vào ROS message
        }
    };

    // Chạy xử lý
    process_markers(ids_5, corners_5, true);
    process_markers(ids_6, corners_6, false);

    // =======================================================
    // 3. IN LOG TERMINAL (MID & DIFF)
    // =======================================================
    if (_enableStdout && (has_center || (has_L && has_R))) {
        if (has_center) {
            std::cout << "[FILT] 5x5 (ID=" << id_center << ") -> 3D: [" 
                      << std::fixed << std::setprecision(4) << center_3d[0] << ", " << center_3d[1] << ", " << center_3d[2] << "]m\n";
        }
        if (has_L && has_R) {
            float mid_x = (L_3d[0] + R_3d[0]) / 2.0f;
            float mid_y = (L_3d[1] + R_3d[1]) / 2.0f;
            float mid_z = (L_3d[2] + R_3d[2]) / 2.0f;

            std::cout << "[FILT] 6x6 (ID=" << id_left << ") -> 3D: [" 
                      << std::fixed << std::setprecision(4) << L_3d[0] << ", " << L_3d[1] << ", " << L_3d[2] << "]m\n";
            std::cout << "[FILT] 6x6 (ID=" << id_right << ") -> 3D: [" 
                      << std::fixed << std::setprecision(4) << R_3d[0] << ", " << R_3d[1] << ", " << R_3d[2] << "]m\n";
            std::cout << "[FILT] MID (6x6) -> 3D: [" 
                      << std::fixed << std::setprecision(4) << mid_x << ", " << mid_y << ", " << mid_z << "]m\n";

            if (has_center) {
                float diff_x = std::abs(center_3d[0] - mid_x) * 1000.0f;
                float diff_y = std::abs(center_3d[1] - mid_y) * 1000.0f;
                float diff_z = std::abs(center_3d[2] - mid_z) * 1000.0f;
                std::cout << ">>> LỆCH MID so với 5x5: dX = " << std::fixed << std::setprecision(1) << diff_x 
                          << "mm | dY = " << std::fixed << std::setprecision(1) << diff_y 
                          << "mm | dZ = " << std::fixed << std::setprecision(1) << diff_z << "mm\n";
            }
        }
        std::cout << "--------------------------------------------------\n";
    }

    return true;

}    
float ArucoDetectorNew::getPlaneDepthRobust(const tmv::DepthFrame& dmsg,
                                            const std::vector<cv::Point2f>& corners) const
{
    // 1. Tính tâm 2D
    cv::Point2f center_pt(0, 0);
    for (const auto& p : corners) center_pt += p;
    center_pt *= 0.25f;

    // Check Safety ROI
    //if (!SAFETY_ROI.contains(cv::Point((int)center_pt.x, (int)center_pt.y))) return 0.0f;

    const int w = static_cast<int>(dmsg.width);
    const int h = static_cast<int>(dmsg.height);
    const int stride_bytes = static_cast<int>(dmsg.stride_bytes > 0 ? dmsg.stride_bytes : w * 2);
    const int stride_u16 = stride_bytes / 2;
    const uint16_t* z16 = dmsg.depth_z16.data();
    const float scale = dmsg.depth_scale;

    const float fx = dmsg.intr.fx;
    const float fy = dmsg.intr.fy;
    const float ppx = dmsg.intr.ppx;
    const float ppy = dmsg.intr.ppy;

    cv::Rect bbox = cv::boundingRect(corners);
    std::vector<cv::Point3f> points_3d;
    
    int step = 3;
    for (int y = bbox.y; y < bbox.y + bbox.height; y += step) {
        if (y < 0 || y >= h) continue;
        const uint16_t* row = z16 + y * stride_u16;

        for (int x = bbox.x; x < bbox.x + bbox.width; x += step) {
            if (x < 0 || x >= w) continue;

            if (cv::pointPolygonTest(corners, cv::Point2f(x, y), false) >= 0) {
                float z = row[x] * scale;
                
                if (z > 0.1f && z < 5.0f) { 
                    float p3d_x, p3d_y, p3d_z;
                    if (deprojectXYZ(p3d_x, p3d_y, p3d_z, static_cast<float>(x), static_cast<float>(y), z, fx, fy, ppx, ppy)) {
                        points_3d.emplace_back(p3d_x, p3d_y, p3d_z);
                    }
                }
            }
        }
    }

    if (points_3d.size() < 30) return 0.0f; 

    // Trộn ngẫu nhiên và khóa max 400 điểm cho RANSAC
    if (points_3d.size() > 400) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(points_3d.begin(), points_3d.end(), g);
        points_3d.resize(400);
    }

    // 2. RANSAC TÌM MẶT PHẲNG CHUẨN
    const int iterations = 30;
    const float threshold = 0.01f; 
    cv::Point3f best_normal;
    float best_d = 0.0f;
    int best_inliers = 0;
    const int early_stop_thresh = static_cast<int>(points_3d.size() * 0.8f);

    for (int i = 0; i < iterations; ++i) {
        cv::Point3f p1 = points_3d[rand() % points_3d.size()];
        cv::Point3f p2 = points_3d[rand() % points_3d.size()];
        cv::Point3f p3 = points_3d[rand() % points_3d.size()];

        cv::Point3f v1 = p2 - p1;
        cv::Point3f v2 = p3 - p1;
        cv::Point3f normal(v1.y * v2.z - v1.z * v2.y,
                           v1.z * v2.x - v1.x * v2.z,
                           v1.x * v2.y - v1.y * v2.x);

        float norm_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        if (norm_len < 1e-6f) continue;
        normal /= norm_len;

        float d = -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z);

        int inliers = 0;
        for (const auto& p : points_3d) {
            float dist = std::abs(normal.x * p.x + normal.y * p.y + normal.z * p.z + d);
            if (dist < threshold) inliers++;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_normal = normal;
            best_d = d;
        }
        if (best_inliers >= early_stop_thresh) break;
    }

    if (best_inliers < points_3d.size() * 0.3) return 0.0f; 
    
    std::vector<cv::Point3f> final_inliers;
    final_inliers.reserve(best_inliers);
    for (const auto& p : points_3d) {
        float dist = std::abs(best_normal.x * p.x + best_normal.y * p.y + best_normal.z * p.z + best_d);
        if (dist < threshold) final_inliers.push_back(p);
    }

    // 3. OLS - LÀM NHẴN MẶT PHẲNG TỚI 0.1mm
    cv::Mat A(final_inliers.size(), 3, CV_32F);
    cv::Mat B(final_inliers.size(), 1, CV_32F);
    for (size_t i = 0; i < final_inliers.size(); ++i) {
        A.at<float>(i, 0) = final_inliers[i].x;
        A.at<float>(i, 1) = final_inliers[i].y;
        A.at<float>(i, 2) = 1.0f;
        B.at<float>(i, 0) = final_inliers[i].z; 
    }

    cv::Mat X_mat;
    float refined_true_z;
    if (cv::solve(A, B, X_mat, cv::DECOMP_SVD)) {
        float a = X_mat.at<float>(0, 0);
        float b = X_mat.at<float>(1, 0);
        float c = X_mat.at<float>(2, 0);

        float local_x = (center_pt.x - ppx) / fx;
        float local_y = (center_pt.y - ppy) / fy;
        
        refined_true_z = c / (1.0f - a * local_x - b * local_y);
    } else {
        return 0.0f; 
    }

    return refined_true_z;
}

// Giữ lại hàm cũ phòng khi cần dùng fallback
float ArucoDetectorNew::depthMedianAt(const tmv::DepthFrame& dmsg,
                                      int u, int v, int half_win) const
{
    const int w = static_cast<int>(dmsg.width);
    const int h = static_cast<int>(dmsg.height);
    const int stride_bytes = static_cast<int>(dmsg.stride_bytes > 0 ? dmsg.stride_bytes : w * 2);

    if (w <= 0 || h <= 0 || stride_bytes <= 0) return 0.0f;
    if (stride_bytes % 2 != 0) return 0.0f;

    const int stride_u16 = stride_bytes / 2;
    if (stride_u16 < w) return 0.0f;
    if (static_cast<int>(dmsg.depth_z16.size()) < h * stride_u16) return 0.0f;

    const uint16_t* z16 = dmsg.depth_z16.data();
    const float scale = dmsg.depth_scale;

    std::vector<float> samples;
    samples.reserve((2 * half_win + 1) * (2 * half_win + 1));

    for (int dy = -half_win; dy <= half_win; ++dy) {
        int yy = clampi(v + dy, 0, h - 1);
        const uint16_t* row = z16 + yy * stride_u16;

        for (int dx = -half_win; dx <= half_win; ++dx) {
            int xx = clampi(u + dx, 0, w - 1);
            uint16_t raw = row[xx];
            if (raw == 0) continue;

            float d = raw * scale;
            if (d > 0.0f && std::isfinite(d)) {
                samples.push_back(d);
            }
        }
    }

    if (samples.empty()) return 0.0f;

    size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    float m = samples[mid];

    if (samples.size() % 2 == 0) {
        std::nth_element(samples.begin(), samples.begin() + mid - 1, samples.end());
        m = 0.5f * (m + samples[mid - 1]);
    }

    return m;
}

bool ArucoDetectorNew::deprojectXYZ(float& X, float& Y, float& Z,
                                    float u, float v,
                                    float depth_m,
                                    float fx, float fy, float ppx, float ppy) const
{
    if (!(depth_m > 0.0f) || !std::isfinite(depth_m)) return false;
    if (!(fx > 0.0f) || !(fy > 0.0f)) return false;

    Z = depth_m;
    X = (u - ppx) * Z / fx;
    Y = (v - ppy) * Z / fy;

    return std::isfinite(X) && std::isfinite(Y) && std::isfinite(Z);
}