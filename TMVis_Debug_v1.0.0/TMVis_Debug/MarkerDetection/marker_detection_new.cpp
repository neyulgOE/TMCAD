#include "MarkerDetection/marker_detection_new.h"
 
// ============================================================================
// [SE3] Noi dung cua se3_pose_optimizer.cpp duoc gop thang vao day (cung ly
// do voi header: gop 1 file de khong phai dong CMakeLists.txt). Toan bo nam
// trong namespace se3opt, khong dung chung ten voi ArucoDetectorNew.
// ============================================================================
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>
 
namespace se3opt {
 
// ============================================================================
// Rodrigues helpers
// ============================================================================
 
// R = I + sin(θ)/θ · [r]× + (1-cos(θ))/θ² · [r]×²
// Với [r]× là ma trận skew-symmetric của r
cv::Mat SE3PoseOptimizer::rodriguesToR(const double r[3])
{
    double theta2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    double theta  = std::sqrt(theta2);
 
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
 
    if (theta < 1e-9) {
        // Khi theta ~ 0: R ≈ I + [r]×
        R.at<double>(0,1) = -r[2]; R.at<double>(0,2) =  r[1];
        R.at<double>(1,0) =  r[2]; R.at<double>(1,2) = -r[0];
        R.at<double>(2,0) = -r[1]; R.at<double>(2,1) =  r[0];
        return R;
    }
 
    double c  = std::cos(theta);
    double s  = std::sin(theta);
    double s_t = s / theta;
    double c1_t2 = (1.0 - c) / theta2;
 
    // K = [r]× / theta (skew của unit vector)
    double kx = r[0]/theta, ky = r[1]/theta, kz = r[2]/theta;
 
    // R = I + sin(θ)·K + (1-cos(θ))·K²
    R.at<double>(0,0) = c + c1_t2*r[0]*r[0];
    R.at<double>(0,1) = c1_t2*r[0]*r[1] - s_t*r[2];
    R.at<double>(0,2) = c1_t2*r[0]*r[2] + s_t*r[1];
 
    R.at<double>(1,0) = c1_t2*r[1]*r[0] + s_t*r[2];
    R.at<double>(1,1) = c + c1_t2*r[1]*r[1];
    R.at<double>(1,2) = c1_t2*r[1]*r[2] - s_t*r[0];
 
    R.at<double>(2,0) = c1_t2*r[2]*r[0] - s_t*r[1];
    R.at<double>(2,1) = c1_t2*r[2]*r[1] + s_t*r[0];
    R.at<double>(2,2) = c + c1_t2*r[2]*r[2];
 
    return R;
}
 
void SE3PoseOptimizer::rToRodrigues(const cv::Mat& R, double r[3])
{
    // Dùng OpenCV Rodrigues để đảm bảo handle singularity đúng
    cv::Mat r_vec;
    cv::Mat R64;
    R.convertTo(R64, CV_64F);
    cv::Rodrigues(R64, r_vec);
    r[0] = r_vec.at<double>(0);
    r[1] = r_vec.at<double>(1);
    r[2] = r_vec.at<double>(2);
}
 
// Jacobian dR/dr: 3 ma trận 3×3, dRdr[k][i][j] = ∂R[i][j]/∂r[k]
// Dùng numerical differentiation với step h = 1e-6 (đủ chính xác cho float32 depth)
void SE3PoseOptimizer::rodriguesToRJacobian(const double r[3], cv::Mat dRdr[3])
{
    const double h = 1e-6;
    cv::Mat R0 = rodriguesToR(r);
    for (int k = 0; k < 3; ++k) {
        double rp[3] = {r[0], r[1], r[2]};
        rp[k] += h;
        cv::Mat Rp = rodriguesToR(rp);
        dRdr[k] = (Rp - R0) * (1.0 / h);
    }
}
 
// ============================================================================
// Marker corners trong marker frame
// ArUco convention: P0=top-left, P1=top-right, P2=bottom-right, P3=bottom-left
// Marker frame: origin ở center, X right, Y down, Z vào camera (ra khỏi mặt marker)
// ============================================================================
std::array<cv::Point3f, 4> SE3PoseOptimizer::markerCorners3D() const
{
    const float h = _cfg.marker_size_m * 0.5f;
    return {{
        cv::Point3f(-h, -h, 0.f),  // P0: top-left
        cv::Point3f( h, -h, 0.f),  // P1: top-right
        cv::Point3f( h,  h, 0.f),  // P2: bottom-right
        cv::Point3f(-h,  h, 0.f),  // P3: bottom-left
    }};
}
 
// ============================================================================
// Transform: điểm từ marker frame sang camera frame
// p_cam = R · p_marker + t
// ============================================================================
cv::Point3f SE3PoseOptimizer::transformPoint(const double xi[6],
                                              const cv::Point3f& p_marker) const
{
    const double* r = xi;
    const double* t = xi + 3;
    cv::Mat R = rodriguesToR(r);
 
    double x = R.at<double>(0,0)*p_marker.x + R.at<double>(0,1)*p_marker.y + R.at<double>(0,2)*p_marker.z + t[0];
    double y = R.at<double>(1,0)*p_marker.x + R.at<double>(1,1)*p_marker.y + R.at<double>(1,2)*p_marker.z + t[1];
    double z = R.at<double>(2,0)*p_marker.x + R.at<double>(2,1)*p_marker.y + R.at<double>(2,2)*p_marker.z + t[2];
    return cv::Point3f(static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z));
}
 
// ============================================================================
// Project 3D → 2D (pinhole, no distortion — giả sử distortion đã compensated)
// ============================================================================
void SE3PoseOptimizer::project(const cv::Point3f& p3d,
                                const CameraIntrinsics& intr,
                                double& u, double& v)
{
    if (p3d.z < 1e-6f) { u = v = 0.0; return; }
    u = intr.fx * (p3d.x / p3d.z) + intr.ppx;
    v = intr.fy * (p3d.y / p3d.z) + intr.ppy;
}
 
// ============================================================================
// Huber weight: ρ'(r²) = min(1, δ/|r|)
// Khi |r| <= δ: weight=1 (L2 thuần), khi |r|>δ: weight<1 (downweight outlier)
// ============================================================================
double SE3PoseOptimizer::huberWeight(double r, double delta) const
{
    double abs_r = std::abs(r);
    if (abs_r < 1e-12) return 1.0;
    return (abs_r <= delta) ? 1.0 : delta / abs_r;
}
 
// ============================================================================
// E1: Reprojection residuals + Jacobian
// res[2k], res[2k+1] = projected - observed (pixel) cho corner k=0..3
// J[2k][0..5], J[2k+1][0..5] = ∂res/∂ξ
// ============================================================================
void SE3PoseOptimizer::computeReprojResiduals(
    const double xi[6],
    const std::array<cv::Point2f, 4>& corners2d,
    const CameraIntrinsics& intr,
    double res[8],
    double J[8][6]) const
{
    auto mc = markerCorners3D();
    const double* r = xi;
    const double* t = xi + 3;
 
    cv::Mat R = rodriguesToR(r);
    cv::Mat dRdr[3];
    if (J) rodriguesToRJacobian(r, dRdr);
 
    for (int k = 0; k < 4; ++k) {
        // p_cam = R * m + t
        double mx = mc[k].x, my = mc[k].y, mz = mc[k].z;
        double px = R.at<double>(0,0)*mx + R.at<double>(0,1)*my + R.at<double>(0,2)*mz + t[0];
        double py = R.at<double>(1,0)*mx + R.at<double>(1,1)*my + R.at<double>(1,2)*mz + t[1];
        double pz = R.at<double>(2,0)*mx + R.at<double>(2,1)*my + R.at<double>(2,2)*mz + t[2];
 
        if (pz < 1e-6) pz = 1e-6;
        double u_proj = intr.fx * px / pz + intr.ppx;
        double v_proj = intr.fy * py / pz + intr.ppy;
 
        res[2*k]   = u_proj - corners2d[k].x;
        res[2*k+1] = v_proj - corners2d[k].y;
 
        if (!J) continue;
 
        // ∂u/∂px = fx/pz, ∂u/∂py = 0, ∂u/∂pz = -fx*px/pz²
        double du_dpx = intr.fx / pz;
        double du_dpz = -intr.fx * px / (pz*pz);
        double dv_dpy = intr.fy / pz;
        double dv_dpz = -intr.fy * py / (pz*pz);
 
        // ∂p/∂r[l] = dRdr[l] * m  (l=0,1,2)
        for (int l = 0; l < 3; ++l) {
            double dpx_drl = dRdr[l].at<double>(0,0)*mx + dRdr[l].at<double>(0,1)*my + dRdr[l].at<double>(0,2)*mz;
            double dpy_drl = dRdr[l].at<double>(1,0)*mx + dRdr[l].at<double>(1,1)*my + dRdr[l].at<double>(1,2)*mz;
            double dpz_drl = dRdr[l].at<double>(2,0)*mx + dRdr[l].at<double>(2,1)*my + dRdr[l].at<double>(2,2)*mz;
 
            J[2*k  ][l] = du_dpx * dpx_drl + du_dpz * dpz_drl;
            J[2*k+1][l] = dv_dpy * dpy_drl + dv_dpz * dpz_drl;
        }
        // ∂p/∂t = I → ∂u/∂t0 = fx/pz, ∂u/∂t1=0, ∂u/∂t2=-fx*px/pz²
        J[2*k  ][3] = du_dpx;   J[2*k  ][4] = 0.0;      J[2*k  ][5] = du_dpz;
        J[2*k+1][3] = 0.0;      J[2*k+1][4] = dv_dpy;   J[2*k+1][5] = dv_dpz;
    }
}
 
// ============================================================================
// E2: Depth residuals + Jacobian (với Huber weighting)
// Residual: plane equation residual = n·p_j + d  (n = R·ẑ, d = -n·t)
// Khi pose T đúng, mọi depth point trên marker phải thỏa n·p_j + d ≈ 0
// ============================================================================
void SE3PoseOptimizer::computeDepthResiduals(
    const double xi[6],
    const std::vector<DepthPoint3D>& pts,
    const CameraIntrinsics& /*intr*/,
    std::vector<double>& res,
    std::vector<std::array<double,6>>& J) const
{
    const double* r = xi;
    const double* t = xi + 3;
 
    cv::Mat R = rodriguesToR(r);
    cv::Mat dRdr[3];
    rodriguesToRJacobian(r, dRdr);
 
    // Normal của marker trong camera frame = R * [0, 0, 1]ᵀ (cột thứ 3 của R)
    double nx = R.at<double>(0,2);
    double ny = R.at<double>(1,2);
    double nz = R.at<double>(2,2);
    // d = -n · t
    double d = -(nx*t[0] + ny*t[1] + nz*t[2]);
 
    // ∂n/∂r[l] = dRdr[l] * [0,0,1]ᵀ = cột thứ 3 của dRdr[l]
    double dn_dr[3][3]; // dn_dr[l][row]
    for (int l = 0; l < 3; ++l) {
        dn_dr[l][0] = dRdr[l].at<double>(0,2);
        dn_dr[l][1] = dRdr[l].at<double>(1,2);
        dn_dr[l][2] = dRdr[l].at<double>(2,2);
    }
 
    const int N = static_cast<int>(pts.size());
    res.resize(N);
    J.resize(N);
 
    for (int j = 0; j < N; ++j) {
        double px = pts[j].p.x;
        double py = pts[j].p.y;
        double pz = pts[j].p.z;
        double w  = std::sqrt(pts[j].weight);  // sqrt vì residual sẽ được squared
 
        // raw residual = n·p + d
        double raw_r = nx*px + ny*py + nz*pz + d;
 
        // Huber reweighting: điều chỉnh effective weight
        double hw = huberWeight(raw_r, _cfg.huber_delta_m);
        double eff_w = w * std::sqrt(hw);
 
        res[j] = eff_w * raw_r;
 
        // ∂(n·p+d)/∂r[l] = (∂n/∂r[l])·p + ∂d/∂r[l]
        // ∂d/∂r[l] = -(∂n/∂r[l]·t) = -(dn_dr[l][0]*t[0]+dn_dr[l][1]*t[1]+dn_dr[l][2]*t[2])
        for (int l = 0; l < 3; ++l) {
            double dn_p = dn_dr[l][0]*px + dn_dr[l][1]*py + dn_dr[l][2]*pz;
            double dn_t = dn_dr[l][0]*t[0] + dn_dr[l][1]*t[1] + dn_dr[l][2]*t[2];
            J[j][l] = eff_w * (dn_p - dn_t);
        }
        // ∂(n·p+d)/∂t[l] = -n[l]
        J[j][3] = -eff_w * nx;
        J[j][4] = -eff_w * ny;
        J[j][5] = -eff_w * nz;
    }
}

// ============================================================================
// [FIX][E2'] Translation-plane-anchor residual — ghim tinh tien t (xi[3..5])
// ve dung mat phang da do robust boi RANSAC+OLS (anchor_normal, anchor_center),
// KHONG dung den rotation r. Khac E2 (point-cloud refit qua GN, de bi bas-
// relief keo lech): E2' dung THANG 1 mat phang da co san (khong tu tao mat
// phang thu 2 doc lap), va chi tac dong len t — Jacobian phan r tuyet doi
// bang 0 nen khong the truc tiep lam lech Roll/Pitch/Yaw (dang duoc tinh
// dung tu E1+E3). Xem ghi chu day du tai Config::w_trans_plane trong header.
//
//   d_anchor = -(anchor_normal · anchor_center)
//   res      = anchor_normal · t + d_anchor   (khoang cach t den mat phang,
//              don vi met — cang gan 0 tuc t cang nam dung tren mat phang)
//   J[0..2]  = 0                              (khong phu thuoc r)
//   J[3..5]  = anchor_normal.x, .y, .z
// ============================================================================
void SE3PoseOptimizer::computeTranslationPlaneAnchorResidual(
    const double xi[6],
    const cv::Point3f& anchor_normal,
    const cv::Point3f& anchor_center,
    double& res,
    double J[6]) const
{
    const double* t = xi + 3;
    double nx = static_cast<double>(anchor_normal.x);
    double ny = static_cast<double>(anchor_normal.y);
    double nz = static_cast<double>(anchor_normal.z);
    double d  = -(nx * anchor_center.x + ny * anchor_center.y + nz * anchor_center.z);

    res = nx * t[0] + ny * t[1] + nz * t[2] + d;

    if (J) {
        J[0] = 0.0; J[1] = 0.0; J[2] = 0.0;
        J[3] = nx;  J[4] = ny;  J[5] = nz;
    }
}
 
// ============================================================================
// E3: Square rigidity residuals + Jacobian
// 5 residuals:
//   r0..3: ‖Pᵢ - Pᵢ₊₁‖ - L  (4 cạnh bằng L)
//   r4:    (P1-P0)·(P2-P1) / L² (góc vuông tại P1)
// Pᵢ = R·mᵢ + t là corners trong camera frame
// ============================================================================
void SE3PoseOptimizer::computeSquareResiduals(
    const double xi[6],
    double res[5],
    double J[5][6]) const
{
    const double* r = xi;
    cv::Mat R = rodriguesToR(r);
    cv::Mat dRdr[3];
    if (J) rodriguesToRJacobian(r, dRdr);
 
    auto mc = markerCorners3D();
    cv::Point3f P[4];
    for (int k = 0; k < 4; ++k) {
        double mx = mc[k].x, my = mc[k].y, mz = mc[k].z;
        P[k].x = static_cast<float>(R.at<double>(0,0)*mx+R.at<double>(0,1)*my+R.at<double>(0,2)*mz + xi[3]);
        P[k].y = static_cast<float>(R.at<double>(1,0)*mx+R.at<double>(1,1)*my+R.at<double>(1,2)*mz + xi[4]);
        P[k].z = static_cast<float>(R.at<double>(2,0)*mx+R.at<double>(2,1)*my+R.at<double>(2,2)*mz + xi[5]);
    }
 
    const double L = _cfg.marker_size_m;
    const double L2 = L * L;
 
    // Edge vectors
    cv::Point3f E[4];
    for (int k = 0; k < 4; ++k) E[k] = P[(k+1)%4] - P[k];
 
    // r0..3: ‖Eₖ‖ - L
    for (int k = 0; k < 4; ++k) {
        double len = std::sqrt(E[k].x*E[k].x + E[k].y*E[k].y + E[k].z*E[k].z);
        res[k] = (len < 1e-9) ? -L : (len - L);
 
        if (!J) continue;
 
        if (len < 1e-9) {
            for (int l = 0; l < 6; ++l) J[k][l] = 0.0;
            continue;
        }
        // ∂‖Eₖ‖/∂ξ = (Eₖ/‖Eₖ‖) · ∂Eₖ/∂ξ
        // Eₖ = P[k+1] - P[k], ∂P[i]/∂r[l] = dRdr[l]*m[i], ∂P[i]/∂t[l] = eₗ
        for (int l = 0; l < 3; ++l) {
            // ∂Eₖ/∂r[l] = dRdr[l]*(m[k+1]-m[k])
            int kn = (k+1)%4;
            double dmx = mc[kn].x - mc[k].x;
            double dmy = mc[kn].y - mc[k].y;
            double dmz = mc[kn].z - mc[k].z;
            double dEx = dRdr[l].at<double>(0,0)*dmx+dRdr[l].at<double>(0,1)*dmy+dRdr[l].at<double>(0,2)*dmz;
            double dEy = dRdr[l].at<double>(1,0)*dmx+dRdr[l].at<double>(1,1)*dmy+dRdr[l].at<double>(1,2)*dmz;
            double dEz = dRdr[l].at<double>(2,0)*dmx+dRdr[l].at<double>(2,1)*dmy+dRdr[l].at<double>(2,2)*dmz;
            J[k][l] = (E[k].x*dEx + E[k].y*dEy + E[k].z*dEz) / len;
        }
        // ∂Eₖ/∂t = 0 (translation không thay đổi edge vector giữa 2 corners)
        J[k][3] = J[k][4] = J[k][5] = 0.0;
    }
 
    // r4: dot(E0, E1) / L² (góc vuông tại P1, E0=P1-P0, E1=P2-P1)
    double dot01 = E[0].x*E[1].x + E[0].y*E[1].y + E[0].z*E[1].z;
    res[4] = dot01 / L2;
 
    if (!J) return;
 
    // ∂(E0·E1)/∂r[l] = (∂E0/∂r[l])·E1 + E0·(∂E1/∂r[l])
    for (int l = 0; l < 3; ++l) {
        // ∂E0/∂r[l] = dRdr[l]*(m1-m0)
        double dm0x = mc[1].x-mc[0].x, dm0y = mc[1].y-mc[0].y, dm0z = mc[1].z-mc[0].z;
        // ∂E1/∂r[l] = dRdr[l]*(m2-m1)
        double dm1x = mc[2].x-mc[1].x, dm1y = mc[2].y-mc[1].y, dm1z = mc[2].z-mc[1].z;
 
        double dE0x = dRdr[l].at<double>(0,0)*dm0x+dRdr[l].at<double>(0,1)*dm0y+dRdr[l].at<double>(0,2)*dm0z;
        double dE0y = dRdr[l].at<double>(1,0)*dm0x+dRdr[l].at<double>(1,1)*dm0y+dRdr[l].at<double>(1,2)*dm0z;
        double dE0z = dRdr[l].at<double>(2,0)*dm0x+dRdr[l].at<double>(2,1)*dm0y+dRdr[l].at<double>(2,2)*dm0z;
 
        double dE1x = dRdr[l].at<double>(0,0)*dm1x+dRdr[l].at<double>(0,1)*dm1y+dRdr[l].at<double>(0,2)*dm1z;
        double dE1y = dRdr[l].at<double>(1,0)*dm1x+dRdr[l].at<double>(1,1)*dm1y+dRdr[l].at<double>(1,2)*dm1z;
        double dE1z = dRdr[l].at<double>(2,0)*dm1x+dRdr[l].at<double>(2,1)*dm1y+dRdr[l].at<double>(2,2)*dm1z;
 
        J[4][l] = (dE0x*E[1].x+dE0y*E[1].y+dE0z*E[1].z
                 + E[0].x*dE1x+E[0].y*dE1y+E[0].z*dE1z) / L2;
    }
    J[4][3] = J[4][4] = J[4][5] = 0.0;
}
 
// ============================================================================
// E5: Temporal regularization — geodesic distance trên SO(3) + Euclidean trên t
// res[0..2] = log(R·Rprev⁻¹)  (rotation residual, Rodrigues vector)
// res[3..5] = t - t_prev       (translation residual, meter)
// ============================================================================
void SE3PoseOptimizer::computeTemporalResiduals(
    const double xi[6],
    const cv::Mat& prev_R,
    const cv::Point3f& prev_t,
    double res[6],
    double J[6][6]) const
{
    const double* r  = xi;
    const double* t  = xi + 3;
 
    cv::Mat R = rodriguesToR(r);
 
    // R_delta = R · prev_R⁻¹ = R · prev_Rᵀ
    cv::Mat prev_R64;
    prev_R.convertTo(prev_R64, CV_64F);
    cv::Mat R64;
    R.convertTo(R64, CV_64F);
    cv::Mat R_delta = R64 * prev_R64.t();
 
    // log(R_delta) = Rodrigues vector của R_delta
    cv::Mat r_delta_vec;
    cv::Rodrigues(R_delta, r_delta_vec);
    res[0] = r_delta_vec.at<double>(0);
    res[1] = r_delta_vec.at<double>(1);
    res[2] = r_delta_vec.at<double>(2);
 
    // Translation residual
    res[3] = t[0] - prev_t.x;
    res[4] = t[1] - prev_t.y;
    res[5] = t[2] - prev_t.z;
 
    if (!J) return;
 
    // Jacobian của log(R·Rprev⁻¹) theo r:
    // Dùng numerical differentiation — đủ chính xác, tránh toán phức tạp Lie algebra
    cv::Mat dRdr[3];
    rodriguesToRJacobian(r, dRdr);
    const double h_inv = 1.0 / 1e-6;
    for (int l = 0; l < 3; ++l) {
        double rp[3] = {r[0],r[1],r[2]};
        rp[l] += 1e-6;
        cv::Mat Rp = rodriguesToR(rp);
        cv::Mat Rp64; Rp.convertTo(Rp64, CV_64F);
        cv::Mat Rdp = Rp64 * prev_R64.t();
        cv::Mat rdp_vec;
        cv::Rodrigues(Rdp, rdp_vec);
        J[0][l] = (rdp_vec.at<double>(0) - res[0]) * h_inv;
        J[1][l] = (rdp_vec.at<double>(1) - res[1]) * h_inv;
        J[2][l] = (rdp_vec.at<double>(2) - res[2]) * h_inv;
    }
    // ∂log(R·Rprev⁻¹)/∂t = 0
    for (int i = 0; i < 3; ++i)
        J[i][3] = J[i][4] = J[i][5] = 0.0;
 
    // ∂(t-t_prev)/∂r = 0, ∂(t-t_prev)/∂t = I
    J[3][0]=J[3][1]=J[3][2]=0.0; J[3][3]=1.0; J[3][4]=0.0; J[3][5]=0.0;
    J[4][0]=J[4][1]=J[4][2]=0.0; J[4][3]=0.0; J[4][4]=1.0; J[4][5]=0.0;
    J[5][0]=J[5][1]=J[5][2]=0.0; J[5][3]=0.0; J[5][4]=0.0; J[5][5]=1.0;
}
 
// ============================================================================
// Accumulate JᵀJ và Jᵀr từ một block residual (n_res × n_param Jacobian)
// JtJ[6][6] += wᵢ · Jᵢᵀ·Jᵢ
// Jtr[6]    += wᵢ · Jᵢᵀ·rᵢ
// ============================================================================
void SE3PoseOptimizer::accumJtJJtr(
    const double* res, const double* J_flat,
    int n_res, int n_param,
    double JtJ[6][6], double Jtr[6])
{
    for (int i = 0; i < n_res; ++i) {
        const double* Ji = J_flat + i * n_param;
        double ri = res[i];
        for (int a = 0; a < n_param; ++a) {
            Jtr[a] += Ji[a] * ri;
            for (int b = a; b < n_param; ++b) {
                JtJ[a][b] += Ji[a] * Ji[b];
            }
        }
    }
    // Fill lower triangle (symmetric)
    for (int a = 1; a < n_param; ++a)
        for (int b = 0; b < a; ++b)
            JtJ[a][b] = JtJ[b][a];
}
 
// ============================================================================
// Solve Levenberg-Marquardt step: (JᵀJ + λ·diag(JᵀJ)) · Δξ = -Jᵀr
// Dùng Cholesky / LDLT qua OpenCV; trả về false nếu singular
// ============================================================================
bool SE3PoseOptimizer::solveLM(double JtJ[6][6], double Jtr[6],
                                 double lambda, double dxi[6])
{
    // Build (JᵀJ + λ·diag) = A
    double A[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            A[i][j] = JtJ[i][j];
    for (int i = 0; i < 6; ++i)
        A[i][i] += lambda * JtJ[i][i];  // Marquardt scaling
 
    cv::Mat Amat(6, 6, CV_64F, A);
    cv::Mat bmat(6, 1, CV_64F);
    for (int i = 0; i < 6; ++i) bmat.at<double>(i) = -Jtr[i];
 
    cv::Mat xmat;
    bool ok = cv::solve(Amat, bmat, xmat, cv::DECOMP_CHOLESKY);
    if (!ok) ok = cv::solve(Amat, bmat, xmat, cv::DECOMP_SVD);
    if (!ok) return false;
 
    for (int i = 0; i < 6; ++i) dxi[i] = xmat.at<double>(i);
    return true;
}
 
// ============================================================================
// Tính tổng cost E(ξ) — không tính Jacobian (chỉ để evaluate)
// [FIX] Chuyen tu free function sang member function (xem ghi chu trong .h)
// ============================================================================
// ============================================================================
// [FIX][E2'] Helper BAO CAO (khong anh huong optimize): tinh RMSE that su
// cua tap diem depth so voi mat phang RANSAC+OLS goc (anchor_normal,
// anchor_center) — khong Huber/confidence weight, giong het cach chinh
// RANSAC+OLS tu bao cao RMSE cua no. Dung de danh gia optimizer co dang
// hoi tu ve dung mat phang da do hay khong, doc lap voi cach GN loop tinh
// residual noi bo (co the bi Huber/confidence lam sai lech y nghia hinh
// hoc thuan tuy). Tra ve -1.0 neu khong the tinh (obs rong hoac thieu anchor).
// ============================================================================
static double computeAnchorPlaneRMSE(const std::vector<se3opt::DepthPoint3D>& pts,
                                      const cv::Point3f& anchor_normal,
                                      const cv::Point3f& anchor_center,
                                      bool has_anchor_normal)
{
    if (!has_anchor_normal || pts.empty()) return -1.0;

    double nx = static_cast<double>(anchor_normal.x);
    double ny = static_cast<double>(anchor_normal.y);
    double nz = static_cast<double>(anchor_normal.z);
    double d  = -(nx * anchor_center.x + ny * anchor_center.y + nz * anchor_center.z);

    double sq = 0.0;
    for (const auto& pt : pts) {
        double r = nx * pt.p.x + ny * pt.p.y + nz * pt.p.z + d;
        sq += r * r;
    }
    return std::sqrt(sq / static_cast<double>(pts.size())) * 1000.0;
}

double SE3PoseOptimizer::computeTotalCost(
    const double xi[6],
    const MarkerObservation& obs,
    const CameraIntrinsics& intr,
    const cv::Mat& prev_R,
    const cv::Point3f& prev_t) const
{
    double cost = 0.0;
    const Config& cfg = _cfg;
 
    // E1
    double res1[8];
    computeReprojResiduals(xi, obs.corners2d, intr, res1, nullptr);
    for (int i = 0; i < 8; ++i) cost += cfg.w_reproj * res1[i]*res1[i];
 
    // E2
    std::vector<double> res2; std::vector<std::array<double,6>> J2;
    computeDepthResiduals(xi, obs.depth_pts, intr, res2, J2);
    for (double r : res2) cost += cfg.w_depth * r*r;

    // E2': translation-plane-anchor (ghim t ve mat phang RANSAC+OLS, khong dung r)
    if (obs.has_anchor_normal) {
        double res_tp;
        computeTranslationPlaneAnchorResidual(xi, obs.anchor_normal, obs.anchor_center, res_tp, nullptr);
        cost += cfg.w_trans_plane * res_tp * res_tp;
    }
 
    // E3
    double res3[5];
    computeSquareResiduals(xi, res3, nullptr);
    for (int i = 0; i < 5; ++i) cost += cfg.w_square * res3[i]*res3[i];
 
    // E5
    if (!prev_R.empty()) {
        double res5[6];
        computeTemporalResiduals(xi, prev_R, prev_t, res5, nullptr);
        for (int i = 0; i < 6; ++i) cost += cfg.w_temporal * res5[i]*res5[i];
    }
 
    return cost;
}
 
// ============================================================================
// optimizeSingle — main optimizer loop (Levenberg-Marquardt)
// ============================================================================
SE3PoseResult SE3PoseOptimizer::optimizeSingle(
    const MarkerObservation& obs,
    const CameraIntrinsics& intr,
    const cv::Mat& init_R,
    const cv::Point3f& init_t,
    const cv::Mat& prev_R,
    const cv::Point3f& prev_t)
{
    SE3PoseResult result;
    if (!obs.valid) return result;
 
    // ---- Initial guess ----
    double xi[6] = {0.0, 0.0, 0.0,
                     static_cast<double>(init_t.x),
                     static_cast<double>(init_t.y),
                     static_cast<double>(init_t.z)};
    if (!init_R.empty()) {
        rToRodrigues(init_R, xi);
    } else {
        // Fallback: assume marker facing camera (z-axis toward camera)
        xi[0] = xi[1] = xi[2] = 0.0;
    }
 
    double lambda = _cfg.lm_lambda_init;
    double prev_cost = computeTotalCost(xi, obs, intr, prev_R, prev_t);
 
    for (int iter = 0; iter < _cfg.max_iter; ++iter) {
        // ---- Accumulate JᵀJ, Jᵀr ----
        double JtJ[6][6] = {};
        double Jtr[6]    = {};
 
        // E1: reprojection
        {
            double res[8]; double J[8][6];
            computeReprojResiduals(xi, obs.corners2d, intr, res, J);
            for (int i = 0; i < 8; ++i) {
                double wr = std::sqrt(_cfg.w_reproj);
                res[i] *= wr;
                for (int j = 0; j < 6; ++j) J[i][j] *= wr;
            }
            accumJtJJtr(res, &J[0][0], 8, 6, JtJ, Jtr);
        }
 
        // E2: depth consistency
        {
            std::vector<double> res; std::vector<std::array<double,6>> J;
            computeDepthResiduals(xi, obs.depth_pts, intr, res, J);
            if (!res.empty()) {
                double wd = std::sqrt(_cfg.w_depth);
                for (size_t i = 0; i < res.size(); ++i) {
                    res[i] *= wd;
                    for (int j = 0; j < 6; ++j) J[i][j] *= wd;
                }
                // Flatten J cho accumJtJJtr
                std::vector<double> J_flat(res.size() * 6);
                for (size_t i = 0; i < res.size(); ++i)
                    for (int j = 0; j < 6; ++j)
                        J_flat[i*6+j] = J[i][j];
                accumJtJJtr(res.data(), J_flat.data(),
                             static_cast<int>(res.size()), 6, JtJ, Jtr);
            }
        }

        // E2': translation-plane-anchor (ghim t ve mat phang RANSAC+OLS,
        // xem Config::w_trans_plane). Jacobian phan r luon = 0 (xem
        // computeTranslationPlaneAnchorResidual), nen term nay KHONG truc
        // tiep dong gop vao cot r cua JtJ/Jtr — chi rang buoc t.
        if (obs.has_anchor_normal) {
            double res_tp; double J_tp[6];
            computeTranslationPlaneAnchorResidual(xi, obs.anchor_normal, obs.anchor_center, res_tp, J_tp);
            double wtp = std::sqrt(_cfg.w_trans_plane);
            res_tp *= wtp;
            for (int j = 0; j < 6; ++j) J_tp[j] *= wtp;
            accumJtJJtr(&res_tp, J_tp, 1, 6, JtJ, Jtr);
        }
 
        // E3: square rigidity
        {
            double res[5]; double J[5][6];
            computeSquareResiduals(xi, res, J);
            double ws = std::sqrt(_cfg.w_square);
            for (int i = 0; i < 5; ++i) {
                res[i] *= ws;
                for (int j = 0; j < 6; ++j) J[i][j] *= ws;
            }
            accumJtJJtr(res, &J[0][0], 5, 6, JtJ, Jtr);
        }
 
        // E5: temporal
        if (!prev_R.empty()) {
            double res[6]; double J[6][6];
            computeTemporalResiduals(xi, prev_R, prev_t, res, J);
            double wt = std::sqrt(_cfg.w_temporal);
            for (int i = 0; i < 6; ++i) {
                res[i] *= wt;
                for (int j = 0; j < 6; ++j) J[i][j] *= wt;
            }
            accumJtJJtr(res, &J[0][0], 6, 6, JtJ, Jtr);
        }
 
        // ---- LM step ----
        double dxi[6];
        bool solved = false;
        for (int lm_try = 0; lm_try < 8; ++lm_try) {
            if (!solveLM(JtJ, Jtr, lambda, dxi)) {
                lambda *= 10.0;
                continue;
            }
            // Step size check
            double dxi_norm = 0.0;
            for (int k = 0; k < 6; ++k) dxi_norm += dxi[k]*dxi[k];
            dxi_norm = std::sqrt(dxi_norm);
 
            double xi_new[6];
            for (int k = 0; k < 6; ++k) xi_new[k] = xi[k] + dxi[k];
 
            double new_cost = computeTotalCost(xi_new, obs, intr, prev_R, prev_t);
 
            if (new_cost < prev_cost) {
                // Bước tốt → accept, giảm damping
                for (int k = 0; k < 6; ++k) xi[k] = xi_new[k];
                lambda = std::max(_cfg.lm_lambda_min, lambda / 3.0);
                prev_cost = new_cost;
                solved = true;
 
                result.iterations = iter + 1;
                if (dxi_norm < _cfg.step_tol) goto converged;
                break;
            } else {
                // Bước xấu → tăng damping, thử lại
                lambda = std::min(_cfg.lm_lambda_max, lambda * 10.0);
            }
        }
 
        if (!solved) break;  // LM bế tắc
    }
 
    converged:
    // [FIX] Truoc day dieu kien "iterations < max_iter" khien 1 marker dung
    // het ngan sach max_iter (van dang cai thien tot, chi chua kip thoa
    // step_tol) bi gan nham converged=false, du khong he "ket". optimizeMulti()
    // ben duoi da dung dung "iterations > 0" — dong bo lai cho nhat quan.
    result.converged = (result.iterations > 0);
    result.final_cost = prev_cost;
 
    // ---- Extract R, t từ xi* ----
    {
        cv::Mat R64 = rodriguesToR(xi);
        R64.convertTo(result.R, CV_32F);
        result.t = cv::Point3f(static_cast<float>(xi[3]),
                                static_cast<float>(xi[4]),
                                static_cast<float>(xi[5]));
    }
 
    // ---- Compute quality metrics ----
    {
        // Reprojection RMSE
        double res1[8]; computeReprojResiduals(xi, obs.corners2d, intr, res1, nullptr);
        double sq = 0.0; for (int i = 0; i < 8; ++i) sq += res1[i]*res1[i];
        result.reproj_rmse_px = std::sqrt(sq / 8.0);
 
        // Depth RMSE
        // [FIX][E2'] Uu tien bao cao RMSE theo mat phang RANSAC+OLS goc
        // (obs.anchor_normal/anchor_center) khi co — day la thuoc do trung
        // thuc chat luong depth thuc te (khong bi Huber/confidence lam
        // lech y nghia hinh hoc). Fallback ve cach cu (dung residual noi
        // bo cua GN, co Huber weight) chi khi obs khong co anchor.
        if (!obs.depth_pts.empty()) {
            double anchorRmse = computeAnchorPlaneRMSE(
                obs.depth_pts, obs.anchor_normal, obs.anchor_center,
                obs.has_anchor_normal);
            if (anchorRmse >= 0.0) {
                result.depth_rmse_mm = anchorRmse;
            } else {
                std::vector<double> res2; std::vector<std::array<double,6>> J2;
                computeDepthResiduals(xi, obs.depth_pts, intr, res2, J2);
                double sq2 = 0.0;
                for (double rv : res2) sq2 += rv*rv;
                result.depth_rmse_mm = std::sqrt(sq2 / res2.size()) * 1000.0;
            }
        }
    }
 
    // ---- Fallback check ----
    if (result.reproj_rmse_px > _cfg.max_acceptable_reproj_px) {
        // Cost quá cao → trả về kết quả init (warm start) thay vì rác
        if (!init_R.empty()) {
            init_R.convertTo(result.R, CV_32F);
            result.t = init_t;
            result.converged = false;
        }
    }
 
    // ---- Roll / Pitch / Yaw từ R* ----
    if (!result.R.empty()) {
        rotMatToRPY(result.R, result.roll_deg, result.pitch_deg, result.yaw_deg);
        result.valid = true;
    }
 
    return result;
}
 
// ============================================================================
// optimizeMulti — optimize 1 rack pose chung cho N markers
// E4 (rigid body): tất cả markers phải nhất quán với 1 pose T
// known_offsets_m[i] = vị trí center marker i trong rack frame
// ============================================================================
SE3PoseResult SE3PoseOptimizer::optimizeMulti(
    const std::vector<MarkerObservation>& obs_list,
    const CameraIntrinsics& intr,
    const std::vector<cv::Point3f>& known_offsets_m,
    const cv::Mat& init_R,
    const cv::Point3f& init_t,
    const cv::Mat& prev_R,
    const cv::Point3f& prev_t)
{
    if (obs_list.empty()) return SE3PoseResult{};
    if (obs_list.size() == 1)
        return optimizeSingle(obs_list[0], intr, init_R, init_t, prev_R, prev_t);
 
    // xi biểu diễn rack frame pose (1 SE(3) duy nhất cho toàn bộ rack)
    double xi[6] = {};
    if (!init_R.empty()) rToRodrigues(init_R, xi);
    xi[3] = init_t.x; xi[4] = init_t.y; xi[5] = init_t.z;
 
    double lambda = _cfg.lm_lambda_init;
    double prev_cost = 1e18;
 
    // Tính cost cho multi-marker: mỗi marker dùng xi + offset_i
    auto computeMultiCost = [&](const double xi_[6]) -> double {
        double cost = 0.0;
        for (size_t mi = 0; mi < obs_list.size(); ++mi) {
            if (!obs_list[mi].valid) continue;
 
            // Pose của marker i = T_rack * T_offset_i
            // T_offset_i: tịnh tiến known_offsets_m[mi] trong rack frame
            // xi_marker = xi nhưng t bù thêm R*offset_i
            cv::Mat R_rack = rodriguesToR(xi_);
            cv::Point3f off = (mi < known_offsets_m.size()) ? known_offsets_m[mi] : cv::Point3f(0,0,0);
            double t_marker[3];
            t_marker[0] = xi_[3] + R_rack.at<double>(0,0)*off.x+R_rack.at<double>(0,1)*off.y+R_rack.at<double>(0,2)*off.z;
            t_marker[1] = xi_[4] + R_rack.at<double>(1,0)*off.x+R_rack.at<double>(1,1)*off.y+R_rack.at<double>(1,2)*off.z;
            t_marker[2] = xi_[5] + R_rack.at<double>(2,0)*off.x+R_rack.at<double>(2,1)*off.y+R_rack.at<double>(2,2)*off.z;
 
            double xi_m[6] = {xi_[0],xi_[1],xi_[2],t_marker[0],t_marker[1],t_marker[2]};
 
            // E1
            double res1[8];
            computeReprojResiduals(xi_m, obs_list[mi].corners2d, intr, res1, nullptr);
            for (int i = 0; i < 8; ++i) cost += _cfg.w_reproj * res1[i]*res1[i];
 
            // E2
            std::vector<double> res2; std::vector<std::array<double,6>> J2;
            computeDepthResiduals(xi_m, obs_list[mi].depth_pts, intr, res2, J2);
            for (double rv : res2) cost += _cfg.w_depth * rv*rv;

            // E2': translation-plane-anchor per-marker (ghim t_marker ve mat
            // phang RANSAC+OLS cua chinh marker do, khong dung r).
            if (obs_list[mi].has_anchor_normal) {
                double res_tp;
                computeTranslationPlaneAnchorResidual(xi_m, obs_list[mi].anchor_normal,
                                                       obs_list[mi].anchor_center, res_tp, nullptr);
                cost += _cfg.w_trans_plane * res_tp * res_tp;
            }
 
            // E3
            double res3[5];
            computeSquareResiduals(xi_m, res3, nullptr);
            for (int i = 0; i < 5; ++i) cost += _cfg.w_square * res3[i]*res3[i];
        }
        // E5
        if (!prev_R.empty()) {
            double res5[6];
            computeTemporalResiduals(xi_, prev_R, prev_t, res5, nullptr);
            for (int i = 0; i < 6; ++i) cost += _cfg.w_temporal * res5[i]*res5[i];
        }
        return cost;
    };
 
    prev_cost = computeMultiCost(xi);
 
    SE3PoseResult result;
    for (int iter = 0; iter < _cfg.max_iter; ++iter) {
        double JtJ[6][6] = {};
        double Jtr[6]    = {};
 
        // Accumulate từng marker (numerical differentiation cho t_marker Jacobian)
        const double h = 1e-6;
        for (size_t mi = 0; mi < obs_list.size(); ++mi) {
            if (!obs_list[mi].valid) continue;
 
            cv::Mat R_rack = rodriguesToR(xi);
            cv::Point3f off = (mi < known_offsets_m.size()) ? known_offsets_m[mi] : cv::Point3f(0,0,0);
            double xi_m[6] = {xi[0],xi[1],xi[2],
                xi[3]+R_rack.at<double>(0,0)*off.x+R_rack.at<double>(0,1)*off.y+R_rack.at<double>(0,2)*off.z,
                xi[4]+R_rack.at<double>(1,0)*off.x+R_rack.at<double>(1,1)*off.y+R_rack.at<double>(1,2)*off.z,
                xi[5]+R_rack.at<double>(2,0)*off.x+R_rack.at<double>(2,1)*off.y+R_rack.at<double>(2,2)*off.z};
 
            // E1 với Jacobian wrt xi (dùng numerical diff để map xi_m→xi)
            double res1[8]; double J1_m[8][6];
            computeReprojResiduals(xi_m, obs_list[mi].corners2d, intr, res1, J1_m);
 
            // Chain rule đơn giản: ∂xi_m/∂xi ~ I (rotation part same, translation bù offset)
            // Đủ chính xác cho GN iteration, refined sau
            double wr = std::sqrt(_cfg.w_reproj);
            for (int i = 0; i < 8; ++i) { res1[i]*=wr; for(int j=0;j<6;j++) J1_m[i][j]*=wr; }
            accumJtJJtr(res1, &J1_m[0][0], 8, 6, JtJ, Jtr);
 
            // E2
            std::vector<double> res2; std::vector<std::array<double,6>> J2;
            computeDepthResiduals(xi_m, obs_list[mi].depth_pts, intr, res2, J2);
            if (!res2.empty()) {
                double wd = std::sqrt(_cfg.w_depth);
                std::vector<double> J2f(res2.size()*6);
                for (size_t i=0;i<res2.size();++i) {
                    res2[i]*=wd;
                    for(int j=0;j<6;j++) J2f[i*6+j]=J2[i][j]*wd;
                }
                accumJtJJtr(res2.data(), J2f.data(), (int)res2.size(), 6, JtJ, Jtr);
            }
 
            // E3
            double res3[5]; double J3[5][6];
            computeSquareResiduals(xi_m, res3, J3);
            double ws = std::sqrt(_cfg.w_square);
            for (int i=0;i<5;++i){res3[i]*=ws; for(int j=0;j<6;j++) J3[i][j]*=ws;}
            accumJtJJtr(res3, &J3[0][0], 5, 6, JtJ, Jtr);

            // E2': translation-plane-anchor per-marker. xi_m[3..5] = xi[3..5]
            // + R_rack*offset (offset khong phu thuoc xi[3..5]) nen
            // d(t_marker)/d(xi[3..5]) = I DUNG CHINH XAC (khong xap xi) —
            // Jacobian tra ve boi computeTranslationPlaneAnchorResidual
            // (wrt xi_m[3..5]) dung thang duoc cho xi[3..5], khong can chain
            // rule nhu E1/E2/E3 o tren. J[0..2]=0 nen khong dinh chinh gi
            // them cho phan rotation cua rack.
            if (obs_list[mi].has_anchor_normal) {
                double res_tp; double J_tp[6];
                computeTranslationPlaneAnchorResidual(xi_m, obs_list[mi].anchor_normal,
                                                       obs_list[mi].anchor_center, res_tp, J_tp);
                double wtp = std::sqrt(_cfg.w_trans_plane);
                res_tp *= wtp;
                for (int j = 0; j < 6; ++j) J_tp[j] *= wtp;
                accumJtJJtr(&res_tp, J_tp, 1, 6, JtJ, Jtr);
            }
        }
 
        // E5
        if (!prev_R.empty()) {
            double res5[6]; double J5[6][6];
            computeTemporalResiduals(xi, prev_R, prev_t, res5, J5);
            double wt = std::sqrt(_cfg.w_temporal);
            for (int i=0;i<6;++i){res5[i]*=wt; for(int j=0;j<6;j++) J5[i][j]*=wt;}
            accumJtJJtr(res5, &J5[0][0], 6, 6, JtJ, Jtr);
        }
 
        // LM step
        bool accepted = false;
        for (int lm_try = 0; lm_try < 8; ++lm_try) {
            double dxi[6];
            if (!solveLM(JtJ, Jtr, lambda, dxi)) { lambda *= 10.0; continue; }
 
            double xi_new[6];
            for (int k=0;k<6;++k) xi_new[k]=xi[k]+dxi[k];
            double new_cost = computeMultiCost(xi_new);
 
            if (new_cost < prev_cost) {
                for (int k=0;k<6;++k) xi[k]=xi_new[k];
                lambda = std::max(_cfg.lm_lambda_min, lambda/3.0);
                prev_cost = new_cost;
                accepted = true;
                result.iterations = iter+1;
                break;
            } else {
                lambda = std::min(_cfg.lm_lambda_max, lambda*10.0);
            }
        }
        if (!accepted) break;
    }
 
    // Extract result
    {
        cv::Mat R64 = rodriguesToR(xi);
        R64.convertTo(result.R, CV_32F);
        result.t = cv::Point3f((float)xi[3],(float)xi[4],(float)xi[5]);
        result.final_cost = prev_cost;
        result.converged = (result.iterations > 0);
        rotMatToRPY(result.R, result.roll_deg, result.pitch_deg, result.yaw_deg);
        result.valid = true;
    }
    return result;
}
 
// ============================================================================
// Build initial guess từ plane normal + center_3d (dùng warm start)
// ============================================================================
bool SE3PoseOptimizer::buildInitialGuess(
    const cv::Point3f& plane_normal,
    const cv::Point3f& center_3d,
    const std::array<cv::Point2f, 4>& corners2d,
    const CameraIntrinsics& intr,
    float marker_size_m,
    cv::Mat& out_R,
    cv::Point3f& out_t)
{
    // n = normalized normal, huong ve camera
    cv::Point3f n = plane_normal;
    float n_len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    if (n_len < 1e-6f) return false;
    n *= (1.f / n_len);
    if (n.z > 0.f) n = -n;  // đảm bảo hướng về camera
 
    // Z axis của marker = -n (normal hướng vào camera = +Z camera)
    cv::Point3f z_marker = -n;
 
    // Estimate X axis từ hướng corner[1]-corner[0] trong 2D (projected)
    // Deproject cả 2 corner để lấy X 3D rough
    // Dùng center depth (center_3d.z) cho cả hai
    float depth = center_3d.z;
    if (depth < 0.05f) return false;
 
    auto deproj = [&](const cv::Point2f& px) -> cv::Point3f {
        float X = (px.x - intr.ppx) * depth / intr.fx;
        float Y = (px.y - intr.ppy) * depth / intr.fy;
        return cv::Point3f(X, Y, depth);
    };
 
    cv::Point3f c0_3d = deproj(corners2d[0]);
    cv::Point3f c1_3d = deproj(corners2d[1]);
    cv::Point3f x_raw = c1_3d - c0_3d;
 
    // Gram-Schmidt: x_marker ⊥ z_marker
    float x_dot_z = x_raw.x*z_marker.x + x_raw.y*z_marker.y + x_raw.z*z_marker.z;
    cv::Point3f x_marker = x_raw - x_dot_z * z_marker;
    float x_len = std::sqrt(x_marker.x*x_marker.x + x_marker.y*x_marker.y + x_marker.z*x_marker.z);
    if (x_len < 1e-6f) {
        // Fallback: chọn trục X tùy ý vuông góc với Z
        if (std::abs(z_marker.x) < 0.9f)
            x_marker = cv::Point3f(1.f,0.f,0.f);
        else
            x_marker = cv::Point3f(0.f,1.f,0.f);
        x_dot_z = x_marker.x*z_marker.x + x_marker.y*z_marker.y + x_marker.z*z_marker.z;
        x_marker = x_marker - x_dot_z * z_marker;
        x_len = std::sqrt(x_marker.x*x_marker.x + x_marker.y*x_marker.y + x_marker.z*x_marker.z);
        if (x_len < 1e-9f) return false;
    }
    x_marker *= (1.f / x_len);
 
    // Y axis = Z × X (right-hand)
    cv::Point3f y_marker(
        z_marker.y*x_marker.z - z_marker.z*x_marker.y,
        z_marker.z*x_marker.x - z_marker.x*x_marker.z,
        z_marker.x*x_marker.y - z_marker.y*x_marker.x);
 
    // Build R = [x_marker | y_marker | z_marker] (columns)
    out_R = cv::Mat(3, 3, CV_32F);
    out_R.at<float>(0,0) = x_marker.x; out_R.at<float>(0,1) = y_marker.x; out_R.at<float>(0,2) = z_marker.x;
    out_R.at<float>(1,0) = x_marker.y; out_R.at<float>(1,1) = y_marker.y; out_R.at<float>(1,2) = z_marker.y;
    out_R.at<float>(2,0) = x_marker.z; out_R.at<float>(2,1) = y_marker.z; out_R.at<float>(2,2) = z_marker.z;
 
    // Re-orthogonalize qua SVD để đảm bảo R ∈ SO(3)
    cv::Mat U, S, Vt;
    cv::SVD::compute(out_R, S, U, Vt);
    cv::Mat R_clean = U * Vt;
    // Ensure det=+1
    if (cv::determinant(R_clean) < 0) {
        U.col(2) *= -1;
        R_clean = U * Vt;
    }
    out_R = R_clean;
 
    // t = center_3d (center marker trong camera frame)
    out_t = center_3d;
 
    return true;
}
 
// ============================================================================
// Roll/Pitch/Yaw từ R (ZYX Euler — thứ tự: roll quanh Z, pitch quanh Y, yaw quanh X)
// Nhưng convention robot thường: RPY = (rotation về X=roll, Y=pitch, Z=yaw)
//
// Dùng convention ZYX (giống code gốc: roll quanh Z-trục-nhìn, pitch quanh X, yaw quanh Y):
//   R = Rz(roll) * Ry(yaw) * Rx(pitch)  — convention cũ của project
//
// Để nhất quán với code gốc:
//   pitch = atan2(ny, -nz) ~ atan2(R[1][2], -R[2][2])
//   roll  = atan2(dy, dx)  ~ atan2(R[1][0], R[0][0]) hoặc atan2(-R[0][1], R[1][1])
//   yaw   = atan2(dz, dx)  ~ atan2(-R[0][2], R[0][0]) tùy convention
//
// Ở đây dùng standard ZYX decomposition để nhất quán với SO(3):
// ============================================================================
void SE3PoseOptimizer::rotMatToRPY(const cv::Mat& R,
                                    double& roll_deg,
                                    double& pitch_deg,
                                    double& yaw_deg)
{
    cv::Mat R64;
    R.convertTo(R64, CV_64F);
 
    // ZYX Euler: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    // pitch = asin(-R[2][0])
    // roll  = atan2(R[2][1], R[2][2])
    // yaw   = atan2(R[1][0], R[0][0])
    //
    // Ánh xạ sang convention của project:
    //   "roll" trong project = quay quanh Z-trục-nhìn (ra/vào camera)
    //   "pitch" = quay quanh X (ngẩng/cúi)
    //   "yaw"   = quay quanh Y (lắc trái/phải)
    //
    // SO(3) ZYX: yaw_zyx = quay quanh Z, pitch_zyx = quay quanh Y, roll_zyx = quay quanh X
    // Map: project_roll=yaw_zyx, project_pitch=roll_zyx, project_yaw=pitch_zyx
 
    double r00 = R64.at<double>(0,0), r01 = R64.at<double>(0,1), r02 = R64.at<double>(0,2);
    double r10 = R64.at<double>(1,0), r11 = R64.at<double>(1,1), r12 = R64.at<double>(1,2);
    double r20 = R64.at<double>(2,0), r21 = R64.at<double>(2,1), r22 = R64.at<double>(2,2);
 
    // Clamp để tránh NaN tại gimbal lock
    double sin_pitch = std::max(-1.0, std::min(1.0, -r20));
    double pitch_zy  = std::asin(sin_pitch);  // ZYX pitch (quay quanh Y)
    double roll_zyx, yaw_zyx;
 
    const double gimbal_thresh = 0.9999;
    if (std::abs(sin_pitch) > gimbal_thresh) {
        // Gimbal lock: roll và yaw không phân biệt được
        roll_zyx = 0.0;
        yaw_zyx  = std::atan2(-r01, r11);
    } else {
        roll_zyx = std::atan2(r21, r22);   // quay quanh X
        yaw_zyx  = std::atan2(r10, r00);   // quay quanh Z
    }
 
    // Map sang convention của project (giống atan2 trong code gốc):
    // project_roll (quay quanh Z-nhìn)  = yaw_zyx
    // project_pitch (ngẩng/cúi, X)      = roll_zyx
    // project_yaw (lắc trái/phải, Y)    = pitch_zyx
    roll_deg  = yaw_zyx  * 180.0 / M_PI;
    pitch_deg = roll_zyx * 180.0 / M_PI;
    yaw_deg   = pitch_zy * 180.0 / M_PI;
 
    // Normalize to [-180, 180]
    auto norm = [](double d) {
        while (d >  180.0) d -= 360.0;
        while (d < -180.0) d += 360.0;
        return d;
    };
    roll_deg  = norm(roll_deg);
    pitch_deg = norm(pitch_deg);
    yaw_deg   = norm(yaw_deg);
}
 
// ============================================================================
// buildMarkerObservation: sample depth points + compute per-point weight
// Weight = 1 / (σ²_local + ε), σ² = variance depth trong cửa sổ var_half_win
// ============================================================================
MarkerObservation buildMarkerObservation(
    int marker_id,
    const std::array<cv::Point2f, 4>& corners2d,
    const DepthFrameView& depth,
    const CameraIntrinsics& intr,
    int step,
    int max_points,
    int var_half_win)
{
    MarkerObservation obs;
    obs.id = marker_id;
    obs.corners2d = corners2d;
 
    if (!depth.z16 || depth.width <= 0 || depth.height <= 0) return obs;
 
    // Convert corners2d sang cv::Point2f vector cho pointPolygonTest
    std::vector<cv::Point2f> poly(corners2d.begin(), corners2d.end());
    cv::Rect bbox = cv::boundingRect(poly);
 
    const int W = depth.width, H = depth.height;
    const int su = depth.stride_u16;
    const float sc = depth.depth_scale;
    const float eps_var = 1e-6f;  // tránh div/0
 
    std::vector<DepthPoint3D> candidates;
    candidates.reserve(512);
 
    for (int y = bbox.y; y < bbox.y + bbox.height; y += step) {
        if (y < 0 || y >= H) continue;
        const uint16_t* row = depth.z16 + y * su;
        for (int x = bbox.x; x < bbox.x + bbox.width; x += step) {
            if (x < 0 || x >= W) continue;
            if (cv::pointPolygonTest(poly, cv::Point2f((float)x,(float)y), false) < 0) continue;
 
            uint16_t raw = row[x];
            if (raw == 0) continue;
            float z = raw * sc;
            if (z < 0.1f || z > 5.0f) continue;
 
            // Deproject
            float px3 = (x - intr.ppx) * z / intr.fx;
            float py3 = (y - intr.ppy) * z / intr.fy;
 
            // Compute local depth variance trong cửa sổ var_half_win
            float sum_d = 0.f, sum_d2 = 0.f;
            int cnt = 0;
            for (int dy = -var_half_win; dy <= var_half_win; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                const uint16_t* rr = depth.z16 + yy * su;
                for (int dx = -var_half_win; dx <= var_half_win; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    uint16_t rv = rr[xx];
                    if (rv == 0) continue;
                    float dv = rv * sc;
                    sum_d  += dv;
                    sum_d2 += dv * dv;
                    cnt++;
                }
            }
            float variance = 0.f;
            if (cnt > 1) {
                float mean = sum_d / cnt;
                variance = sum_d2 / cnt - mean * mean;
                if (variance < 0.f) variance = 0.f;
            }
            // weight = 1 / (variance + eps) — normalize sau
            float weight = 1.0f / (variance + eps_var);
 
            candidates.push_back(DepthPoint3D{cv::Point3f(px3,py3,z), weight});
        }
    }
 
    if (candidates.empty()) return obs;
 
    // Subsample nếu quá nhiều
    if (static_cast<int>(candidates.size()) > max_points) {
        std::mt19937 rng(42);
        std::shuffle(candidates.begin(), candidates.end(), rng);
        candidates.resize(max_points);
    }
 
    // Normalize weights về [0,1]
    float max_w = 0.f;
    for (auto& c : candidates) max_w = std::max(max_w, c.weight);
    if (max_w > 0.f) {
        for (auto& c : candidates) c.weight /= max_w;
    }
 
    obs.depth_pts = std::move(candidates);
    obs.valid = true;
    return obs;
}
 
} // namespace se3opt
// ============================================================================
// [SE3] Het phan gop tu se3_pose_optimizer.cpp
// ============================================================================
 
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
 
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
 
// ============================================================================
// [ORIENTATION] Helper tu do (khong phai member) — giu giong main11.cpp
// ============================================================================
namespace {
 
inline double normalizeAngleDeg180(double deg) {
    while (deg > 180.0)  deg -= 360.0;
    while (deg < -180.0) deg += 360.0;
    return deg;
}
 
} // namespace
 
ArucoDetectorNew::ArucoDetectorNew(const std::string& name)
    : Plugin(name)
{
    loadParams();
 
    _dict = cv::aruco::getPredefinedDictionary(
        static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(_dictId));
 
    _params = cv::aruco::DetectorParameters::create();
    _params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
 
    _clahe = cv::createCLAHE(_claheClipLimit, cv::Size(_claheTileSize, _claheTileSize));
}
 
ArucoDetectorNew::~ArucoDetectorNew()
{
    _running.store(false);
}
 
void ArucoDetectorNew::loadParams()
{
    _dictId        = static_cast<int>(cv::aruco::DICT_6X6_100);
    _loopSleepMs   = 100;
    _maxSyncDiffMs = 50;
    _roiHalfWin    = 3;
    _enableXYZ     = true;
    _enableStdout  = true;
 
    _maxSyncDiffNs = static_cast<uint64_t>(_maxSyncDiffMs) * 1000000ULL;
 
    _enableSpatialFilter = true;
    _spatialGaussKernel  = 3;
    _spatialGaussSigma   = 0.8;
    _enableCLAHE         = false;
    _claheClipLimit      = 2.0;
    _claheTileSize       = 8;
 
    _enableTemporalFilter = true;
    _temporalEmaAlpha     = 0.4;
    _offsetX = -0.0325f;
    _offsetY =  0.0f;
    _offsetZ =  0.0f;
 
    // [SE3] Kich thuoc canh marker vat ly (m) — dung cho square-rigidity
    // (E3) va init-guess cua optimizer. Phai khop marker_size_m thuc te.
    _markerSizeM = 0.06f;
 
    // [SE3][E4] TUY CHON: neu rack co >=2 marker gan co dinh, do vi tri tam
    // tung marker trong rack frame (m) roi dien vao day de bat optimizeMulti
    // (rigid-body constraint, 12DOF -> 6DOF). Vi du 2 marker cach nhau 30cm,
    // dat doc theo truc X cua rack:
    //   _knownRackOffsetsM[1] = cv::Point3f(-0.15f, 0.f, 0.f);
    //   _knownRackOffsetsM[2] = cv::Point3f( 0.15f, 0.f, 0.f);
    // Neu khong dien gi, code tu dong fallback ve pipeline per-marker.
    _knownRackOffsetsM.clear();
}
 
void ArucoDetectorNew::run()
{
    _running.store(true);
    if (_enableStdout)
    {
        std::cout << "[ArucoNew][Running] Dict=6x6_100 + Pre-detection Filtering + Orientation(RPY)\n"
                  << "  Spatial filter : "
                  << (_enableSpatialFilter
                         ? ("Gaussian k=" + std::to_string(_spatialGaussKernel) +
                            " sigma=" + std::to_string(_spatialGaussSigma))
                         : "OFF") << "\n"
                  << "  CLAHE          : " << (_enableCLAHE ? "ON" : "OFF") << "\n"
                  << "  Temporal filter: "
                  << (_enableTemporalFilter
                         ? ("EMA alpha=" + std::to_string(_temporalEmaAlpha))
                         : "OFF") << "\n"
                  << "  Orientation    : "
                  << (_enableOrientFilter
                         ? ("EMA alpha=" + std::to_string(_orientFilterAlpha))
                         : "raw (no filter)") << "\n";
    }
 
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
        tryBuildDetection(det);
        std::this_thread::sleep_for(std::chrono::milliseconds(_loopSleepMs));
    }
}
 
cv::Mat ArucoDetectorNew::applySpatialFilter(const cv::Mat& gray) const
{
    cv::Mat result = gray;
 
    if (_enableCLAHE && _clahe)
    {
        cv::Mat clahe_out;
        _clahe->apply(result, clahe_out);
        result = clahe_out;
    }
 
    if (_enableSpatialFilter && _spatialGaussKernel >= 3)
    {
        const int k = (_spatialGaussKernel % 2 == 0)
                      ? _spatialGaussKernel + 1
                      : _spatialGaussKernel;
        cv::Mat blurred;
        cv::GaussianBlur(result, blurred,
                         cv::Size(k, k), _spatialGaussSigma, _spatialGaussSigma);
        result = blurred;
    }
 
    return result;
}
 
cv::Mat ArucoDetectorNew::applyTemporalFilter(const cv::Mat& gray)
{
    if (!_enableTemporalFilter) return gray;
 
    const double alpha = std::max(0.05, std::min(_temporalEmaAlpha, 1.0));
 
    cv::Mat gray_f32;
    gray.convertTo(gray_f32, CV_32F);
 
    if (!_grayEmaInit || _grayEmaAccum.size() != gray_f32.size())
    {
        _grayEmaAccum = gray_f32.clone();
        _grayEmaInit  = true;
    }
    else
    {
        cv::addWeighted(_grayEmaAccum, 1.0 - alpha,
                        gray_f32,      alpha,
                        0.0, _grayEmaAccum);
    }
 
    cv::Mat out;
    _grayEmaAccum.convertTo(out, CV_8U);
    return out;
}
 
// ============================================================================
// [ORIENTATION] EMA scalar filter dung chung cho roll/pitch/yaw.
// ============================================================================
void ArucoDetectorNew::applyScalarFilter(ScalarFilterState& s, double raw_value,
                                         double alpha, bool is_angle_deg)
{
    if (!s.valid) {
        s.valid = true;
        s.value = raw_value;
        return;
    }
 
    double delta = raw_value - s.value;
    if (is_angle_deg) delta = normalizeAngleDeg180(delta);
 
    double updated = s.value + alpha * delta;
    if (is_angle_deg) updated = normalizeAngleDeg180(updated);
 
    s.value = updated;
}
 
// ============================================================================
// [ORIENTATION] Tinh Roll/Pitch/Yaw tu danh sach marker sample hop le trong
// 1 frame. Port truc tiep tu compute_orientation_from_markers() cua
// main11.cpp — KHONG mang theo apply_depth_correction()/offset cua main11,
// center_3d cua sample la toa do camera-frame CHUA hieu chuan.
// ============================================================================
void ArucoDetectorNew::computeOrientationFromMarkers(const std::vector<MarkerPoseSample>& samples,
                                                     bool has_joint_normal,
                                                     const cv::Point3f& joint_normal,
                                                     float joint_rmse_mm)
{
    _orientState.roll_valid  = false;
    _orientState.pitch_valid = false;
    _orientState.yaw_valid   = false;
 
    std::vector<MarkerPoseSample> valid;
    valid.reserve(samples.size());
    for (const auto& s : samples) {
        if (s.has_normal) valid.push_back(s);
    }
 
    if (valid.empty()) return;
 
    bool   roll_ok  = false;
    double roll_raw = 0.0;
    bool   yaw_ok    = false;
    double yaw_raw   = 0.0;
 
    bool        have_pair = false;
    cv::Point3f pair_left_3d, pair_right_3d;
 
    if (valid.size() >= 2) {
        // Sort theo X 3D (camera frame) de xac dinh trai/phai.
        std::sort(valid.begin(), valid.end(),
                  [](const MarkerPoseSample& a, const MarkerPoseSample& b) {
                      return a.center_3d.x < b.center_3d.x;
                  });
 
        const MarkerPoseSample& left  = valid.front();
        const MarkerPoseSample& right = valid.back();
        have_pair     = true;
        pair_left_3d  = left.center_3d;
        pair_right_3d = right.center_3d;
 
        const double dx = static_cast<double>(right.center_3d.x - left.center_3d.x);
        const double dy = static_cast<double>(right.center_3d.y - left.center_3d.y);
        const double dz = static_cast<double>(right.center_3d.z - left.center_3d.z);
 
        // ---- Roll (quay quanh Z, truc nhin) — tu (dX, dY), bo qua dZ ----
        if (std::abs(dx) > 1e-6 || std::abs(dy) > 1e-6) {
            roll_raw = std::atan2(dy, dx) * 180.0 / M_PI;
            roll_raw = normalizeAngleDeg180(roll_raw);
            roll_ok  = std::isfinite(roll_raw);
        }
 
        // ---- Yaw (quay quanh Y, lac trai/phai) tu 2 marker 3D ----
        // theta = atan2(dZ, dX): doi xung +/-theta, khong bi asymmetry cua
        // depth sensor, khong nhay voi pitch != 0 nhu yaw tu normal.
        {
            const double d_xz = std::sqrt(dx * dx + dz * dz);
            if (d_xz > 1e-6) {
                yaw_raw = std::atan2(dz, dx) * 180.0 / M_PI;
                yaw_raw = normalizeAngleDeg180(yaw_raw);
                yaw_ok  = std::isfinite(yaw_raw);
            }
        }
    }
 
    // ---- Pitch (quanh X, ngan/cup) tu plane normal ----
    //
    // [PITCH FIX] Weighted-average normal tung marker (normal_sum ben duoi)
    // co baseline khong gian chi vai cm (dien tich 1 marker) -> sai so uoc
    // luong do nghieng (a,b) lon, gay pitch troi du Z tung marker rat on
    // dinh (xem phan tich log thuc te). Joint-plane normal (fit tren toan
    // bo diem cua ca 2 marker, baseline ~300mm) chinh xac hon nhieu cho
    // cung ly do baseline ma Roll/Yaw dang huong loi. Uu tien dung joint
    // normal khi co (>=2 marker); fallback ve weighted-average khi chi co
    // 1 marker (khong the tinh joint fit).
    cv::Point3f normal_sum(0.f, 0.f, 0.f);
    {
        const double eps = 0.1;
        for (const auto& s : valid) {
            double w = 1.0 / (static_cast<double>(s.rmse_mm) + eps);
            normal_sum.x += static_cast<float>(w) * s.normal.x;
            normal_sum.y += static_cast<float>(w) * s.normal.y;
            normal_sum.z += static_cast<float>(w) * s.normal.z;
        }
    }
    double normal_len = std::sqrt(static_cast<double>(normal_sum.x) * normal_sum.x +
                                  static_cast<double>(normal_sum.y) * normal_sum.y +
                                  static_cast<double>(normal_sum.z) * normal_sum.z);
 
    bool        pitch_ok = false;
    double      pitch_raw = 0.0;
    cv::Point3f effective_normal(0.f, 0.f, -1.f);
    bool        used_joint_normal = false;
 
    if (has_joint_normal) {
        const float jl = std::sqrt(joint_normal.x * joint_normal.x +
                                    joint_normal.y * joint_normal.y +
                                    joint_normal.z * joint_normal.z);
        if (jl > 1e-6f) {
            effective_normal  = joint_normal * (1.0f / jl);
            used_joint_normal = true;
        }
    }
    if (!used_joint_normal && normal_len > 1e-9) {
        effective_normal = cv::Point3f(static_cast<float>(normal_sum.x / normal_len),
                                        static_cast<float>(normal_sum.y / normal_len),
                                        static_cast<float>(normal_sum.z / normal_len));
    }
    const bool have_effective_normal = used_joint_normal || (normal_len > 1e-9);
 
    if (have_effective_normal) {
        // N(theta) = (0, sin(theta), -cos(theta)) -> theta = atan2(ny, -nz)
        pitch_raw = std::atan2(static_cast<double>(effective_normal.y),
                               -static_cast<double>(effective_normal.z)) * 180.0 / M_PI;
        pitch_ok  = std::isfinite(pitch_raw);
 
        if (_enableStdout && pitch_ok) {
            std::cout << "[ArucoNew][ORIENT-DEBUG] pitch_raw=" << std::fixed << std::setprecision(3)
                      << pitch_raw << "deg source=" << (used_joint_normal ? "joint-plane(accum)" : "per-marker-avg")
                      << " normal=(" << std::setprecision(4)
                      << effective_normal.x << "," << effective_normal.y << "," << effective_normal.z << ")";
            if (used_joint_normal) {
                std::cout << " joint_rmse_lastfit=" << std::setprecision(3) << joint_rmse_mm << "mm";
            }
            std::cout << "\n";
        }
    }
 
    // ---- Yaw fallback tu normal (chi 1 marker) ----
    if (!yaw_ok && normal_len > 1e-9) {
        const double nx = normal_sum.x / normal_len;
        const double nz = normal_sum.z / normal_len;
        yaw_raw = std::atan2(nx, -nz) * 180.0 / M_PI;
        yaw_raw = normalizeAngleDeg180(yaw_raw);
        yaw_ok  = std::isfinite(yaw_raw);
    }
 
    // ---- [RACK FRAME] Gram-Schmidt tu effective_normal (uu tien joint-plane
    // khi co) + duong noi 2 tam marker ----
    _orientState.rack_frame_valid = false;
    if (have_pair && pitch_ok && have_effective_normal) {
        const cv::Point3f z_axis = effective_normal;
 
        cv::Point3f v = pair_right_3d - pair_left_3d;
        float v_dot_z = v.x * z_axis.x + v.y * z_axis.y + v.z * z_axis.z;
        cv::Point3f v_proj = v - v_dot_z * z_axis;
        float v_proj_len = std::sqrt(v_proj.x * v_proj.x + v_proj.y * v_proj.y + v_proj.z * v_proj.z);
 
        if (v_proj_len > 1e-6f) {
            cv::Point3f x_axis = v_proj * (1.0f / v_proj_len);
            cv::Point3f y_axis(x_axis.y * z_axis.z - x_axis.z * z_axis.y,
                                x_axis.z * z_axis.x - x_axis.x * z_axis.z,
                                x_axis.x * z_axis.y - x_axis.y * z_axis.x);
 
            _orientState.rack_origin = (pair_left_3d + pair_right_3d) * 0.5f;
            _orientState.rack_x_axis = x_axis;
            _orientState.rack_y_axis = y_axis;
            _orientState.rack_z_axis = z_axis;
            _orientState.rack_frame_valid = true;
        }
    }
 
    if (roll_ok) {
        if (_enableOrientFilter) {
            applyScalarFilter(_rollFilter, roll_raw, _orientFilterAlpha, true);
            _orientState.roll_deg = _rollFilter.value;
        } else {
            _orientState.roll_deg = roll_raw;
        }
        _orientState.roll_valid = true;
    }
 
    if (yaw_ok) {
        if (_enableOrientFilter) {
            applyScalarFilter(_yawFilter, yaw_raw, _orientFilterAlpha, true);
            _orientState.yaw_deg = _yawFilter.value;
        } else {
            _orientState.yaw_deg = yaw_raw;
        }
        _orientState.yaw_valid = true;
    }
 
    if (pitch_ok) {
        if (_enableOrientFilter) {
            applyScalarFilter(_pitchFilter, pitch_raw, _orientFilterAlpha, true);
            _orientState.pitch_deg = _pitchFilter.value;
        } else {
            _orientState.pitch_deg = pitch_raw;
        }
        _orientState.pitch_valid = true;
    }
}
 
// ============================================================================
// [SE3][E4] Ap ket qua optimizeMulti() (R*,t* rigid cho ca rack) thang vao
// _orientState. R* da truc giao (SO(3)) nen KHONG can Gram-Schmidt/atan2
// tren vector noi 2 tam marker nhu pipeline cu — 3 cot cua R* CHINH LA 3
// truc rack, va rotMatToRPY() cua optimizer dung dung convention project
// (roll=quanh Z, pitch=quanh X, yaw=quanh Y) nen doc thang duoc.
// Van giu EMA scalar filter o day de mem hoa qua cac frame marker bi mat
// (rack pose luc do fallback ve legacy path, EMA giup chuyen tiep khong giat).
// ============================================================================
void ArucoDetectorNew::applySE3RackOrientation(const se3opt::SE3PoseResult& rack)
{
    double roll_raw  = rack.roll_deg;
    double pitch_raw = rack.pitch_deg;
    double yaw_raw   = rack.yaw_deg;
 
    if (_enableOrientFilter) {
        applyScalarFilter(_rollFilter,  roll_raw,  _orientFilterAlpha, true);
        applyScalarFilter(_pitchFilter, pitch_raw, _orientFilterAlpha, true);
        applyScalarFilter(_yawFilter,   yaw_raw,   _orientFilterAlpha, true);
        _orientState.roll_deg  = _rollFilter.value;
        _orientState.pitch_deg = _pitchFilter.value;
        _orientState.yaw_deg   = _yawFilter.value;
    } else {
        _orientState.roll_deg  = roll_raw;
        _orientState.pitch_deg = pitch_raw;
        _orientState.yaw_deg   = yaw_raw;
    }
    _orientState.roll_valid  = true;
    _orientState.pitch_valid = true;
    _orientState.yaw_valid   = true;
 
    // Rack axes = 3 cot cua R* (da truc giao san, khong can Gram-Schmidt)
    cv::Mat R32;
    rack.R.convertTo(R32, CV_32F);
    _orientState.rack_x_axis = cv::Point3f(R32.at<float>(0, 0), R32.at<float>(1, 0), R32.at<float>(2, 0));
    _orientState.rack_y_axis = cv::Point3f(R32.at<float>(0, 1), R32.at<float>(1, 1), R32.at<float>(2, 1));
    _orientState.rack_z_axis = cv::Point3f(R32.at<float>(0, 2), R32.at<float>(1, 2), R32.at<float>(2, 2));
    _orientState.rack_origin = rack.t;
    _orientState.rack_frame_valid = true;
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
 
    const int w      = static_cast<int>(rmsg.width);
    const int h      = static_cast<int>(rmsg.height);
    const int stride = static_cast<int>(rmsg.stride_bytes > 0 ? rmsg.stride_bytes : w * 3);
 
    if (w <= 0 || h <= 0) return false;
    if (static_cast<int>(rmsg.bgr8.size()) < h * stride) return false;
 
    cv::Mat bgrView(h, w, CV_8UC3, (void*)rmsg.bgr8.data(), stride);
    cv::Mat bgr = bgrView.clone();
 
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
 
    const cv::Mat grayDetect = applyTemporalFilter(applySpatialFilter(gray));
 
    // ── Detect với dict 6x6 duy nhất ─────────────────────────────────────
    auto dict_6x6 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
 
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(grayDetect, dict_6x6, corners, ids, _params);
 
    out = tmv::MarkerDetections{};
    out.header.pub_nsec  = dns;
    out.header.data_nsec = dns;
    out.width  = static_cast<uint32_t>(w);
    out.height = static_cast<uint32_t>(h);
    out.valid  = true;
 
    // [ORIENTATION] Thu thap sample (center_3d chua hieu chuan + normal) cho
    // moi marker hop le trong frame nay, dung de tinh Roll/Pitch/Yaw sau khi
    // xu ly xong tat ca marker.
    std::vector<MarkerPoseSample> orientSamples;
    // [PITCH FIX] Corners cua cac marker co plane-fit hop le, dung lam input
    // cho getJointPlaneNormal() (fit 1 mat phang chung, baseline lon hon).
    std::vector<std::vector<cv::Point2f>> validCornersForJoint;
 
    if (ids.empty()) {
        // Khong marker nao -> khong tinh lai orientation, giu gia tri EMA
        // truoc do nhung dat *_valid=false (giong hanh vi main11).
        computeOrientationFromMarkers(orientSamples);
        return true;
    }
 
    const float fx  = dmsg.intr.fx;
    const float fy  = dmsg.intr.fy;
    const float ppx = dmsg.intr.ppx;
    const float ppy = dmsg.intr.ppy;
 
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (corners[i].size() != 4) continue;
 
        cv::Point2f c(0.0f, 0.0f);
        for (int k = 0; k < 4; ++k) c += corners[i][k];
        c *= 0.25f;
 
        tmv::Marker mk;
        mk.id = ids[i];
 
        const int cu  = clampi(static_cast<int>(std::lround(c.x)), 0, w - 1);
        const int cvv = clampi(static_cast<int>(std::lround(c.y)), 0, h - 1);
 
        mk.center.u = static_cast<uint32_t>(cu);
        mk.center.v = static_cast<uint32_t>(cvv);
 
        auto fillUV = [&](tmv::PixelUV& p, const cv::Point2f& pt) {
            p.u = static_cast<uint32_t>(
                clampi(static_cast<int>(std::lround(pt.x)), 0, w - 1));
            p.v = static_cast<uint32_t>(
                clampi(static_cast<int>(std::lround(pt.y)), 0, h - 1));
        };
        fillUV(mk.c0, corners[i][0]);
        fillUV(mk.c1, corners[i][1]);
        fillUV(mk.c2, corners[i][2]);
        fillUV(mk.c3, corners[i][3]);
 
        if (_enableXYZ)
        {
            ArucoPlaneDepthResult result = getPlaneDepthRobust(dmsg, corners[i], mk.id);
            float X = 0.0f, Y = 0.0f, Z = 0.0f;
            bool ok = false;
 
            if (result.valid && result.depth_m > 0.0f)
                ok = deprojectXYZ(X, Y, Z,
                                  static_cast<float>(cu), static_cast<float>(cvv),
                                  result.depth_m, fx, fy, ppx, ppy);
 
            mk.has_xyz = ok;
 
            // [ORIENTATION] Dung X/Y/Z CHUA offset (truoc khi cong _offsetX/Y/Z)
            // lam center_3d cho sample orientation — KHONG mang offset hieu
            // chuan cua plugin vao phep tinh goc.
            if (ok && result.has_normal)
            {
                // ------------------------------------------------------
                // [SE3] Thay vi dung thang plane-fit (result.normal / X,Y,Z)
                // lam sample orientation, chay optimizeSingle() de refine:
                // reprojection (E1) + depth-plane (E2) + square-rigidity
                // (E3) + temporal warm-start tu frame truoc (E5) cung luc.
                // plane-fit cu (result) chi con dung de:
                //   (a) tao init-guess (buildInitialGuess), va
                //   (b) fallback neu optimizer khong hoi tu / obs invalid.
                // ------------------------------------------------------
                se3opt::CameraIntrinsics se3intr{fx, fy, ppx, ppy};
                std::array<cv::Point2f, 4> cornersArr{
                    corners[i][0], corners[i][1], corners[i][2], corners[i][3]};
 
                se3opt::DepthFrameView depthView;
                depthView.z16         = dmsg.depth_z16.data();
                depthView.width       = w;
                depthView.height      = h;
                depthView.stride_u16  = static_cast<int>(
                    dmsg.stride_bytes > 0 ? (dmsg.stride_bytes / 2) : (dmsg.width));
                depthView.depth_scale = dmsg.depth_scale;
 
                se3opt::MarkerObservation obs = se3opt::buildMarkerObservation(
                    mk.id, cornersArr, depthView, se3intr);

                // [FIX][E2'] Ghim mat phang RANSAC+OLS da co (result.normal,
                // center tai (X,Y,Z) CHUA offset hieu chuan, khop he quy
                // chieu voi t cua optimizer) lam anchor cho translation-
                // plane-anchor residual. Xem Config::w_trans_plane va
                // computeTranslationPlaneAnchorResidual.
                if (result.has_normal && Z > 0.05f) {
                    obs.anchor_normal    = result.normal;
                    obs.anchor_center    = cv::Point3f(X, Y, Z);
                    obs.has_anchor_normal = true;
                }

                // [DEBUG][E2'-CHECK] Xac nhan flag anchor thuc su duoc bat
                // luc runtime — tranh lap lai tinh huong nghi ngo build cu/
                // flag khong active nhu truoc. Co the xoa sau khi da xac
                // nhan on dinh qua vai lan chay thu.
                if (_enableStdout) {
                    std::cout << "[SE3][ANCHOR-CHECK] id=" << mk.id
                              << " has_anchor_normal=" << obs.has_anchor_normal
                              << " w_trans_plane=" << _se3Opt.getConfig().w_trans_plane
                              << " anchor_normal=(" << obs.anchor_normal.x << ","
                              << obs.anchor_normal.y << "," << obs.anchor_normal.z << ")\n";
                }
 
                cv::Mat     init_R;
                cv::Point3f init_t;
                bool haveInit = se3opt::SE3PoseOptimizer::buildInitialGuess(
                    result.normal, cv::Point3f(X, Y, Z), cornersArr, se3intr,
                    _markerSizeM, init_R, init_t);
 
                cv::Mat     prevR;
                cv::Point3f prevT{0.f, 0.f, 0.f};
                auto itPrev = _prevPoseByMarker.find(mk.id);
                if (itPrev != _prevPoseByMarker.end() && itPrev->second.valid) {
                    prevR = itPrev->second.R;
                    prevT = itPrev->second.t;
                }
 
                se3opt::SE3PoseResult se3res;
                if (obs.valid && haveInit) {
                    se3res = _se3Opt.optimizeSingle(obs, se3intr, init_R, init_t, prevR, prevT);
                }
 
                if (se3res.valid) {
                    // Cache lam warm-start cho frame ke tiep (E5)
                    _prevPoseByMarker[mk.id] = PrevPoseSE3{se3res.R, se3res.t, true};
 
                    // Normal marker = cot Z cua R* (marker frame Z huong ve
                    // camera, dung convention giong result.normal cu)
                    cv::Point3f se3Normal(se3res.R.at<float>(0, 2),
                                          se3res.R.at<float>(1, 2),
                                          se3res.R.at<float>(2, 2));
                    float qualityRmseMm = (se3res.depth_rmse_mm > 0.0)
                                         ? static_cast<float>(se3res.depth_rmse_mm)
                                         : result.rmse_mm;
 
                    _normalAccum[mk.id].push(se3Normal, qualityRmseMm);
                    cv::Point3f use_normal = _normalAccum[mk.id].empty()
                                            ? se3Normal
                                            : _normalAccum[mk.id].averaged();
 
                    MarkerPoseSample sample;
                    sample.id         = mk.id;
                    sample.center_px  = c;
                    sample.center_3d  = se3res.t;   // SE3-refined center (chua offset)
                    sample.normal     = use_normal;
                    sample.has_normal = true;
                    sample.rmse_mm    = qualityRmseMm;
                    orientSamples.push_back(sample);
 
                    validCornersForJoint.push_back(corners[i]);
 
                    // Dung X/Y/Z tinh chinh (chinh xac hon Z-only plane-fit
                    // cu) cho output mk.x/y/z ben duoi
                    X = se3res.t.x; Y = se3res.t.y; Z = se3res.t.z;
 
                    if (_enableStdout) {
                        std::cout << "[ArucoNew][SE3][ID:" << mk.id << "] iters="
                                  << se3res.iterations << " converged=" << se3res.converged
                                  << " reproj_rmse=" << std::fixed << std::setprecision(3)
                                  << se3res.reproj_rmse_px << "px depth_rmse="
                                  << se3res.depth_rmse_mm << "mm (raw plane-fit was "
                                  << result.rmse_mm << "mm)";
                        // [DEBUG] Neu SE3 lam depth-fit TE HON han plane-fit
                        // thuan tuy -> nghi ngo bas-relief ambiguity / huber
                        // qua chat, dang de reprojection lan at depth. Xem
                        // ghi chu tai Config::huber_delta_m trong header.
                        if (se3res.depth_rmse_mm > 2.0 * result.rmse_mm && result.rmse_mm > 0.1f) {
                            std::cout << "  <-- CANH BAO: depth-fit te hon RANSAC+OLS goc"
                                         " ~" << (se3res.depth_rmse_mm / result.rmse_mm)
                                      << "x, kiem tra huber_delta_m / can bang weight E1-E2";
                        }
                        std::cout << "\n";
                    }
                } else {
                    // ---- Fallback: optimizer khong hoi tu / obs invalid ----
                    // -> dung plane-fit cu (result) nhu code goc.
                    _normalAccum[mk.id].push(result.normal, result.rmse_mm);
                    cv::Point3f use_normal = _normalAccum[mk.id].empty()
                                            ? result.normal
                                            : _normalAccum[mk.id].averaged();
 
                    MarkerPoseSample sample;
                    sample.id         = mk.id;
                    sample.center_px  = c;
                    sample.center_3d  = cv::Point3f(X, Y, Z);
                    sample.normal     = use_normal;
                    sample.has_normal = true;
                    sample.rmse_mm    = result.rmse_mm;
                    orientSamples.push_back(sample);
 
                    validCornersForJoint.push_back(corners[i]);
                }
            }
 
            if (ok)
            {
                X += _offsetX;
                Y += _offsetY;
                Z += _offsetZ;
                mk.x = X; mk.y = Y; mk.z = Z;
 
                if (_enableStdout)
                {
                    std::cout << "[ArucoNew] 6x6 ID=" << mk.id
                              << " -> 3D: ["
                              << std::fixed << std::setprecision(4)
                              << X << ", " << Y << ", " << Z << "]m"
                              << "  RMSE=" << std::setprecision(2) << result.rmse_mm << "mm\n";
                }
            }
        }
 
        out.markers.push_back(mk);
    }
 
    // [PITCH FIX] Neu co >=2 marker hop le, fit 1 mat phang chung tren toan
    // bo diem cua ca hai marker de lay normal baseline lon (~300mm) dung
    // rieng cho Pitch — thay vi weighted-average 2 normal fit rieng biet
    // (baseline chi vai cm, sai so do nghieng lon hon nhieu).
    //
    // [PITCH FIX v2] Baseline gop chi tang theo truc X (2 marker dat ngang
    // hang) -> chi giup nx (Roll/Yaw-related) on dinh, KHONG giup ny (nguon
    // cua Pitch) vi khong co them baseline theo truc Y. De giam bot nhieu
    // con lai cua ny (mot phan do RANSAC ngau nhien moi frame), tich luy
    // joint-normal qua 12 frame (weighted 1/rmse, giong normal tung marker)
    // truoc khi dua vao tinh Pitch.
    ArucoPlaneDepthResult jointResult;
    bool        use_joint_normal = false;
    cv::Point3f joint_normal_for_orient(0.f, 0.f, -1.f);
    float       joint_rmse_for_orient = 0.0f;
 
    if (validCornersForJoint.size() >= 2) {
        jointResult = getJointPlaneNormal(dmsg, validCornersForJoint);
        if (jointResult.valid && jointResult.has_normal) {
            _jointNormalAccum.push(jointResult.normal, jointResult.rmse_mm);
 
            joint_normal_for_orient = _jointNormalAccum.empty()
                                     ? jointResult.normal
                                     : _jointNormalAccum.averaged();
            joint_rmse_for_orient   = jointResult.rmse_mm;
            use_joint_normal        = true;
        }
    }
 
    // ------------------------------------------------------------------
    // [SE3][E4] Neu da co calibration _knownRackOffsetsM va >=2 marker hop
    // le trong frame nay, chay optimizeMulti() de toi uu 1 SE(3) DUY NHAT
    // cho ca rack (rigid-body constraint giua cac marker) — thay hoan toan
    // atan2(dy,dx)/atan2(dz,dx) tren vector noi 2 tam + Gram-Schmidt thu
    // cong. R* tra ve da truc giao san nen Roll/Pitch/Yaw + 3 truc rack
    // deu doc thang tu R*, khong con cross-axis coupling.
    // ------------------------------------------------------------------
    bool                   se3RackValid = false;
    se3opt::SE3PoseResult  rackResult;
 
    if (!_knownRackOffsetsM.empty() && orientSamples.size() >= 2) {
        se3opt::CameraIntrinsics se3intr{fx, fy, ppx, ppy};
        std::vector<se3opt::MarkerObservation> obsList;
        std::vector<cv::Point3f>               offsets;
        obsList.reserve(orientSamples.size());
        offsets.reserve(orientSamples.size());
 
        for (size_t si = 0; si < ids.size(); ++si) {
            if (corners[si].size() != 4) continue;
            auto itOff = _knownRackOffsetsM.find(ids[si]);
            if (itOff == _knownRackOffsetsM.end()) continue;
 
            std::array<cv::Point2f, 4> cornersArr{
                corners[si][0], corners[si][1], corners[si][2], corners[si][3]};
 
            se3opt::DepthFrameView depthView;
            depthView.z16         = dmsg.depth_z16.data();
            depthView.width       = w;
            depthView.height      = h;
            depthView.stride_u16  = static_cast<int>(
                dmsg.stride_bytes > 0 ? (dmsg.stride_bytes / 2) : (dmsg.width));
            depthView.depth_scale = dmsg.depth_scale;
 
            se3opt::MarkerObservation obs = se3opt::buildMarkerObservation(
                ids[si], cornersArr, depthView, se3intr);
            if (!obs.valid) continue;

            // [FIX][E2'] Ghim anchor cho rack path — dung CUNG nguon
            // orientSamples da tinh o vong lap chinh phia tren (sample.normal/
            // center_3d, co the la SE3-refined-per-marker neu marker do da
            // hoi tu qua optimizeSingle() truoc khi vao rack, hoac raw
            // RANSAC+OLS neu fallback). Khong tinh lai plane-fit o day de
            // tranh trung cong — nhat quan voi cach lam o single-marker path.
            {
                auto itS = std::find_if(orientSamples.begin(), orientSamples.end(),
                    [&](const MarkerPoseSample& s) { return s.id == ids[si]; });
                if (itS != orientSamples.end() && itS->has_normal
                    && itS->center_3d.z > 0.05f) {
                    obs.anchor_normal     = itS->normal;
                    obs.anchor_center     = itS->center_3d;
                    obs.has_anchor_normal = true;
                }
            }
 
            obsList.push_back(std::move(obs));
            offsets.push_back(itOff->second);
        }
 
        if (obsList.size() >= 2) {
            cv::Mat     prevR = _prevRackPose.valid ? _prevRackPose.R : cv::Mat();
            cv::Point3f prevT = _prevRackPose.valid ? _prevRackPose.t : cv::Point3f(0.f, 0.f, 0.f);
 
            cv::Mat     init_R = prevR;   // warm-start = frame truoc neu co
            cv::Point3f init_t = _prevRackPose.valid
                                ? _prevRackPose.t
                                : (orientSamples.front().center_3d + orientSamples.back().center_3d) * 0.5f;
 
            rackResult = _se3Opt.optimizeMulti(obsList, se3intr, offsets,
                                               init_R, init_t, prevR, prevT);
            if (rackResult.valid) {
                _prevRackPose = PrevPoseSE3{rackResult.R, rackResult.t, true};
                se3RackValid  = true;
 
                if (_enableStdout) {
                    std::cout << "[ArucoNew][SE3][RACK] markers=" << obsList.size()
                              << " iters=" << rackResult.iterations
                              << " converged=" << rackResult.converged
                              << " reproj_rmse=" << std::fixed << std::setprecision(3)
                              << rackResult.reproj_rmse_px << "px\n";
                }
            }
        }
    }
 
    // [ORIENTATION] Cap nhat Roll/Pitch/Yaw sau khi da xu ly xong tat ca
    // marker trong frame. Neu <1 marker co normal -> giu gia tri cu, *_valid=false.
    if (se3RackValid) {
        applySE3RackOrientation(rackResult);
    } else {
        computeOrientationFromMarkers(orientSamples,
                                      use_joint_normal,
                                      joint_normal_for_orient,
                                      joint_rmse_for_orient);
    }
 
    if (_enableStdout)
    {
        if (_orientState.roll_valid || _orientState.pitch_valid || _orientState.yaw_valid)
        {
            std::cout << "[ArucoNew][ORIENT] roll="
                      << (_orientState.roll_valid ? _orientState.roll_deg
                                                   : std::numeric_limits<double>::quiet_NaN())
                      << "deg pitch="
                      << (_orientState.pitch_valid ? _orientState.pitch_deg
                                                    : std::numeric_limits<double>::quiet_NaN())
                      << "deg yaw="
                      << (_orientState.yaw_valid ? _orientState.yaw_deg
                                                  : std::numeric_limits<double>::quiet_NaN())
                      << "deg\n";
        }
        else
        {
            std::cout << "[ArucoNew][ORIENT] N/A (can >=1 marker co normal hop le)\n";
        }
 
        if (!ids.empty())
            std::cout << "--------------------------------------------------\n";
    }
 
    return true;
}
 
// ============================================================================
// [PLANE DEPTH + NORMAL] Fit mat phang tren cac diem 3D lay tu polygon
// marker (grid step=3, cap 400 diem — giu nguyen sampling budget cho
// RK3588). Tra ve Z tam marker (OLS refine, x2 pass loai outlier >2σ) VA
// plane normal (camera frame) dung cho Roll/Pitch — port tu
// get_aruco_plane_depth_robust() cua main11.cpp.
// ============================================================================
ArucoPlaneDepthResult ArucoDetectorNew::getPlaneDepthRobust(const tmv::DepthFrame& dmsg,
                                            const std::vector<cv::Point2f>& corners,
                                            int marker_id) const
{
    ArucoPlaneDepthResult res;
 
    cv::Point2f center_pt(0, 0);
    for (const auto& p : corners) center_pt += p;
    center_pt *= 0.25f;
 
    const int w            = static_cast<int>(dmsg.width);
    const int h            = static_cast<int>(dmsg.height);
    const int stride_bytes = static_cast<int>(dmsg.stride_bytes > 0 ? dmsg.stride_bytes : w * 2);
    const int stride_u16   = stride_bytes / 2;
    const uint16_t* z16    = dmsg.depth_z16.data();
    const float scale      = dmsg.depth_scale;
 
    const float fx  = dmsg.intr.fx;
    const float fy  = dmsg.intr.fy;
    const float ppx = dmsg.intr.ppx;
    const float ppy = dmsg.intr.ppy;
 
    cv::Rect bbox = cv::boundingRect(corners);
    std::vector<cv::Point3f> points_3d;
 
    const int step = 3;
    for (int y = bbox.y; y < bbox.y + bbox.height; y += step) {
        if (y < 0 || y >= h) continue;
        const uint16_t* row = z16 + y * stride_u16;
        for (int x = bbox.x; x < bbox.x + bbox.width; x += step) {
            if (x < 0 || x >= w) continue;
            if (cv::pointPolygonTest(corners,
                                     cv::Point2f(static_cast<float>(x),
                                                 static_cast<float>(y)),
                                     false) >= 0)
            {
                float z = row[x] * scale;
                if (z > 0.1f && z < 5.0f) {
                    float p3d_x, p3d_y, p3d_z;
                    if (deprojectXYZ(p3d_x, p3d_y, p3d_z,
                                     static_cast<float>(x), static_cast<float>(y),
                                     z, fx, fy, ppx, ppy))
                        points_3d.emplace_back(p3d_x, p3d_y, p3d_z);
                }
            }
        }
    }
 
    if (static_cast<int>(points_3d.size()) < 30) return res;
 
    if (points_3d.size() > 400) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(points_3d.begin(), points_3d.end(), g);
        points_3d.resize(400);
    }
 
    // 1. RANSAC tim mat phang tho
    const int   iterations = 30;
    const float threshold  = 0.01f;
    cv::Point3f best_normal;
    float best_d       = 0.0f;
    int   best_inliers = 0;
    const int early_stop_thresh = static_cast<int>(points_3d.size() * 0.8f);
 
    for (int it = 0; it < iterations; ++it) {
        cv::Point3f p1 = points_3d[rand() % points_3d.size()];
        cv::Point3f p2 = points_3d[rand() % points_3d.size()];
        cv::Point3f p3 = points_3d[rand() % points_3d.size()];
 
        cv::Point3f v1 = p2 - p1;
        cv::Point3f v2 = p3 - p1;
        cv::Point3f normal(v1.y * v2.z - v1.z * v2.y,
                           v1.z * v2.x - v1.x * v2.z,
                           v1.x * v2.y - v1.y * v2.x);
 
        float norm_len = std::sqrt(normal.x*normal.x +
                                   normal.y*normal.y +
                                   normal.z*normal.z);
        if (norm_len < 1e-6f) continue;
        normal /= norm_len;
 
        float d = -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z);
 
        int inliers = 0;
        for (const auto& p : points_3d) {
            float dist = std::abs(normal.x * p.x + normal.y * p.y +
                                  normal.z * p.z + d);
            if (dist < threshold) inliers++;
        }
 
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_normal  = normal;
            best_d       = d;
        }
        if (best_inliers >= early_stop_thresh) break;
    }
 
    if (best_inliers < static_cast<int>(points_3d.size() * 0.3f)) return res;
 
    std::vector<cv::Point3f> final_inliers;
    final_inliers.reserve(best_inliers);
    for (const auto& p : points_3d) {
        float dist = std::abs(best_normal.x * p.x + best_normal.y * p.y +
                              best_normal.z * p.z + best_d);
        if (dist < threshold) final_inliers.push_back(p);
    }
 
    // 2. OLS lan 1 — lam nhan mat phang (z = a*x + b*y + c)
    cv::Mat A(static_cast<int>(final_inliers.size()), 3, CV_32F);
    cv::Mat B(static_cast<int>(final_inliers.size()), 1, CV_32F);
    for (int i = 0; i < static_cast<int>(final_inliers.size()); ++i) {
        A.at<float>(i, 0) = final_inliers[i].x;
        A.at<float>(i, 1) = final_inliers[i].y;
        A.at<float>(i, 2) = 1.0f;
        B.at<float>(i, 0) = final_inliers[i].z;
    }
 
    cv::Mat X_mat;
    float a, b, c, refined_true_z;
    if (!cv::solve(A, B, X_mat, cv::DECOMP_SVD)) return res;
 
    a = X_mat.at<float>(0, 0);
    b = X_mat.at<float>(1, 0);
    c = X_mat.at<float>(2, 0);
 
    {
        const float local_x = (center_pt.x - ppx) / fx;
        const float local_y = (center_pt.y - ppy) / fy;
        const float denom   = 1.0f - a * local_x - b * local_y;
        if (std::abs(denom) < 1e-6f) return res;
        refined_true_z = c / denom;
    }
 
    // 3. RMSE lan 1 — dung lam nguong loai outlier cho OLS lan 2
    float sq_err = 0.0f;
    for (const auto& p : final_inliers) {
        float z_fit = a * p.x + b * p.y + c;
        float err   = p.z - z_fit;
        sq_err += err * err;
    }
    float rmse = std::sqrt(sq_err / final_inliers.size());
 
    // 4. OLS lan 2 — loai outlier >2σ roi refit, normal sach hon (giong main11)
    {
        const float thresh2 = 2.0f * rmse;
        std::vector<cv::Point3f> inliers2;
        inliers2.reserve(final_inliers.size());
        for (const auto& p : final_inliers) {
            float z_fit = a * p.x + b * p.y + c;
            if (std::abs(p.z - z_fit) <= thresh2) inliers2.push_back(p);
        }
 
        if (inliers2.size() >= 10) {
            cv::Mat A2(static_cast<int>(inliers2.size()), 3, CV_32F);
            cv::Mat B2(static_cast<int>(inliers2.size()), 1, CV_32F);
            for (size_t i = 0; i < inliers2.size(); ++i) {
                A2.at<float>(static_cast<int>(i), 0) = inliers2[i].x;
                A2.at<float>(static_cast<int>(i), 1) = inliers2[i].y;
                A2.at<float>(static_cast<int>(i), 2) = 1.0f;
                B2.at<float>(static_cast<int>(i), 0) = inliers2[i].z;
            }
            cv::Mat X2;
            if (cv::solve(A2, B2, X2, cv::DECOMP_SVD)) {
                a = X2.at<float>(0, 0);
                b = X2.at<float>(1, 0);
                c = X2.at<float>(2, 0);
 
                const float local_x2 = (center_pt.x - ppx) / fx;
                const float local_y2 = (center_pt.y - ppy) / fy;
                const float denom2   = 1.0f - a * local_x2 - b * local_y2;
                if (std::abs(denom2) > 1e-6f) {
                    refined_true_z = c / denom2;
 
                    float sq2 = 0.0f;
                    for (const auto& p : inliers2) {
                        float e = p.z - (a * p.x + b * p.y + c);
                        sq2 += e * e;
                    }
                    rmse = std::sqrt(sq2 / inliers2.size());
                    final_inliers.swap(inliers2);
                }
            }
        }
    }
 
    res.depth_m  = refined_true_z;
    res.rmse_mm  = rmse * 1000.0f;
    res.inliers  = static_cast<int>(final_inliers.size());
    res.valid    = true;
 
    // 5. Plane normal tu he so OLS (a, b, -1), huong ve phia camera de Roll/
    // Pitch co quy uoc dau nhat quan.
    {
        cv::Point3f n(a, b, -1.0f);
        float n_len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (n_len > 1e-9f) {
            n *= (1.0f / n_len);
            if (n.z > 0.0f) n = -n;
            res.normal     = n;
            res.has_normal = true;
        }
    }
 
    if (_enableStdout)
    {
        std::cout << "[ArucoNew][ID:" << marker_id << "] [LSQ-RANSAC] Inliers: " << res.inliers
                  << " | RMSE: " << std::fixed << std::setprecision(3) << res.rmse_mm << "mm"
                  << " | Z_plane(OLS): " << std::fixed << std::setprecision(4) << res.depth_m << "m"
                  << " | normal=(" << std::setprecision(4)
                  << res.normal.x << "," << res.normal.y << "," << res.normal.z << ")\n";
    }
 
    return res;
}
 
// ============================================================================
// [PITCH FIX] Fit 1 mat phang duy nhat tren toan bo diem 3D gop tu tat ca
// marker trong cornersList — cung thuat toan RANSAC(30 vong)+OLS x2 pass
// nhu getPlaneDepthRobust(), chi khac o buoc lay mau: gop diem tu nhieu
// polygon marker thay vi 1 polygon. Baseline khong gian cua tap diem tang
// tu vai cm (1 marker) len xap xi khoang cach giua cac marker (~300mm voi
// cau hinh 2-marker hien tai) -> giam manh sai so uoc luong do nghieng
// (a,b) cua mat phang, day chinh la nguyen nhan pitch troi trong khi Z
// tung marker (tinh rieng, khong doi) van rat on dinh.
// ============================================================================
ArucoPlaneDepthResult ArucoDetectorNew::getJointPlaneNormal(const tmv::DepthFrame& dmsg,
                                    const std::vector<std::vector<cv::Point2f>>& cornersList) const
{
    ArucoPlaneDepthResult res;
    if (cornersList.size() < 2) return res;
 
    const int w            = static_cast<int>(dmsg.width);
    const int h            = static_cast<int>(dmsg.height);
    const int stride_bytes = static_cast<int>(dmsg.stride_bytes > 0 ? dmsg.stride_bytes : w * 2);
    const int stride_u16   = stride_bytes / 2;
    const uint16_t* z16    = dmsg.depth_z16.data();
    const float scale      = dmsg.depth_scale;
 
    const float fx  = dmsg.intr.fx;
    const float fy  = dmsg.intr.fy;
    const float ppx = dmsg.intr.ppx;
    const float ppy = dmsg.intr.ppy;
 
    std::vector<cv::Point3f> points_3d;
    const int step = 3;
 
    for (const auto& corners : cornersList) {
        if (corners.size() != 4) continue;
        cv::Rect bbox = cv::boundingRect(corners);
        for (int y = bbox.y; y < bbox.y + bbox.height; y += step) {
            if (y < 0 || y >= h) continue;
            const uint16_t* row = z16 + y * stride_u16;
            for (int x = bbox.x; x < bbox.x + bbox.width; x += step) {
                if (x < 0 || x >= w) continue;
                if (cv::pointPolygonTest(corners,
                                         cv::Point2f(static_cast<float>(x),
                                                     static_cast<float>(y)),
                                         false) >= 0)
                {
                    float z = row[x] * scale;
                    if (z > 0.1f && z < 5.0f) {
                        float p3d_x, p3d_y, p3d_z;
                        if (deprojectXYZ(p3d_x, p3d_y, p3d_z,
                                         static_cast<float>(x), static_cast<float>(y),
                                         z, fx, fy, ppx, ppy))
                            points_3d.emplace_back(p3d_x, p3d_y, p3d_z);
                    }
                }
            }
        }
    }
 
    if (static_cast<int>(points_3d.size()) < 30) return res;
 
    // Gioi han tong so diem gop lai ve cung muc (400) nhu fit 1-marker de
    // giu ngan sach CPU khong doi — mat do diem tren 1 marker giam nhung
    // do trai rong khong gian (yeu to quyet dinh do chinh xac normal) tang.
    if (points_3d.size() > 400) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(points_3d.begin(), points_3d.end(), g);
        points_3d.resize(400);
    }
 
    const int   iterations = 30;
    const float threshold  = 0.01f;
    cv::Point3f best_normal;
    float best_d       = 0.0f;
    int   best_inliers = 0;
    const int early_stop_thresh = static_cast<int>(points_3d.size() * 0.8f);
 
    for (int it = 0; it < iterations; ++it) {
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
            best_normal  = normal;
            best_d       = d;
        }
        if (best_inliers >= early_stop_thresh) break;
    }
 
    if (best_inliers < static_cast<int>(points_3d.size() * 0.3f)) return res;
 
    std::vector<cv::Point3f> final_inliers;
    final_inliers.reserve(best_inliers);
    for (const auto& p : points_3d) {
        float dist = std::abs(best_normal.x * p.x + best_normal.y * p.y + best_normal.z * p.z + best_d);
        if (dist < threshold) final_inliers.push_back(p);
    }
 
    cv::Mat A(static_cast<int>(final_inliers.size()), 3, CV_32F);
    cv::Mat B(static_cast<int>(final_inliers.size()), 1, CV_32F);
    for (int i = 0; i < static_cast<int>(final_inliers.size()); ++i) {
        A.at<float>(i, 0) = final_inliers[i].x;
        A.at<float>(i, 1) = final_inliers[i].y;
        A.at<float>(i, 2) = 1.0f;
        B.at<float>(i, 0) = final_inliers[i].z;
    }
 
    cv::Mat X_mat;
    if (!cv::solve(A, B, X_mat, cv::DECOMP_SVD)) return res;
 
    float a = X_mat.at<float>(0, 0);
    float b = X_mat.at<float>(1, 0);
    float c = X_mat.at<float>(2, 0);
 
    float sq_err = 0.0f;
    for (const auto& p : final_inliers) {
        float err = p.z - (a * p.x + b * p.y + c);
        sq_err += err * err;
    }
    float rmse = std::sqrt(sq_err / final_inliers.size());
 
    // OLS lan 2 — loai outlier >2σ roi refit, giong getPlaneDepthRobust()
    {
        const float thresh2 = 2.0f * rmse;
        std::vector<cv::Point3f> inliers2;
        inliers2.reserve(final_inliers.size());
        for (const auto& p : final_inliers) {
            float z_fit = a * p.x + b * p.y + c;
            if (std::abs(p.z - z_fit) <= thresh2) inliers2.push_back(p);
        }
 
        if (inliers2.size() >= 10) {
            cv::Mat A2(static_cast<int>(inliers2.size()), 3, CV_32F);
            cv::Mat B2(static_cast<int>(inliers2.size()), 1, CV_32F);
            for (size_t i = 0; i < inliers2.size(); ++i) {
                A2.at<float>(static_cast<int>(i), 0) = inliers2[i].x;
                A2.at<float>(static_cast<int>(i), 1) = inliers2[i].y;
                A2.at<float>(static_cast<int>(i), 2) = 1.0f;
                B2.at<float>(static_cast<int>(i), 0) = inliers2[i].z;
            }
            cv::Mat X2;
            if (cv::solve(A2, B2, X2, cv::DECOMP_SVD)) {
                a = X2.at<float>(0, 0);
                b = X2.at<float>(1, 0);
                c = X2.at<float>(2, 0);
 
                float sq2 = 0.0f;
                for (const auto& p : inliers2) {
                    float e = p.z - (a * p.x + b * p.y + c);
                    sq2 += e * e;
                }
                rmse = std::sqrt(sq2 / inliers2.size());
                final_inliers.swap(inliers2);
            }
        }
    }
 
    res.rmse_mm  = rmse * 1000.0f;
    res.inliers  = static_cast<int>(final_inliers.size());
    res.valid    = true;
 
    cv::Point3f n(a, b, -1.0f);
    float n_len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (n_len > 1e-9f) {
        n *= (1.0f / n_len);
        if (n.z > 0.0f) n = -n;
        res.normal     = n;
        res.has_normal = true;
    }
 
    if (_enableStdout && res.has_normal)
    {
        std::cout << "[ArucoNew][JOINT-PLANE] markers=" << cornersList.size()
                  << " points=" << points_3d.size()
                  << " inliers=" << res.inliers
                  << " | RMSE: " << std::fixed << std::setprecision(3) << res.rmse_mm << "mm"
                  << " | normal=(" << std::setprecision(4)
                  << res.normal.x << "," << res.normal.y << "," << res.normal.z << ")\n";
    }
 
    return res;
}
 
float ArucoDetectorNew::depthMedianAt(const tmv::DepthFrame& dmsg,
                                      int u, int v, int half_win) const
{
    const int w            = static_cast<int>(dmsg.width);
    const int h            = static_cast<int>(dmsg.height);
    const int stride_bytes = static_cast<int>(dmsg.stride_bytes > 0 ? dmsg.stride_bytes : w * 2);
 
    if (w <= 0 || h <= 0 || stride_bytes <= 0) return 0.0f;
    if (stride_bytes % 2 != 0) return 0.0f;
 
    const int stride_u16 = stride_bytes / 2;
    if (stride_u16 < w) return 0.0f;
    if (static_cast<int>(dmsg.depth_z16.size()) < h * stride_u16) return 0.0f;
 
    const uint16_t* z16   = dmsg.depth_z16.data();
    const float     scale = dmsg.depth_scale;
 
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>((2 * half_win + 1) * (2 * half_win + 1)));
 
    for (int dy = -half_win; dy <= half_win; ++dy) {
        const uint16_t* row = z16 + clampi(v + dy, 0, h - 1) * stride_u16;
        for (int dx = -half_win; dx <= half_win; ++dx) {
            uint16_t raw = row[clampi(u + dx, 0, w - 1)];
            if (raw == 0) continue;
            float d = raw * scale;
            if (d > 0.0f && std::isfinite(d)) samples.push_back(d);
        }
    }
 
    if (samples.empty()) return 0.0f;
 
    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    float m = samples[mid];
 
    if (samples.size() % 2 == 0 && mid > 0) {
        std::nth_element(samples.begin(), samples.begin() + mid - 1, samples.end());
        m = 0.5f * (m + samples[mid - 1]);
    }
 
    return m;
}
 
bool ArucoDetectorNew::deprojectXYZ(float& X, float& Y, float& Z,
                                    float u, float v, float depth_m,
                                    float fx, float fy, float ppx, float ppy) const
{
    if (!(depth_m > 0.0f) || !std::isfinite(depth_m)) return false;
    if (!(fx > 0.0f) || !(fy > 0.0f)) return false;
    Z = depth_m;
    X = (u - ppx) * Z / fx;
    Y = (v - ppy) * Z / fy;
    return std::isfinite(X) && std::isfinite(Y) && std::isfinite(Z);
}