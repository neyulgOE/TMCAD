#pragma once
 
#include <atomic>
#include <vector>
#include <cstdint>
#include <string>
 
#include <map>
#include <deque>
#include <numeric>
#include <sstream>
#include <chrono>
 
#include <array>
 
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
 
#include "ClientAPI/Plugin.h"
#include "common/DataModel.h"
 
// ============================================================================
// [SE3] Noi dung cua se3_pose_optimizer.h duoc gop thang vao day (thay vi
// include file rieng) de khoi phai sua CMakeLists.txt / lo include path.
// Toan bo optimizer nam trong namespace se3opt, doc lap voi ArucoDetectorNew
// ben duoi — co the tach ra file rieng lai bat cu luc nao neu can.
// ============================================================================
 
// ============================================================================
// SE(3) Pose Optimizer cho ArUco marker với depth camera
//
// Thay thế pipeline atan2 hiện tại (roll/pitch/yaw tính từ 3 nguồn độc lập)
// bằng một bài toán tối ưu duy nhất trên SE(3): tìm T=(R,t) minimize tổng
// residual từ tất cả observation đồng thời.
//
// Cost function E(ξ) gồm 5 term:
//   E1 — 2D reprojection của 4 corners qua (R,t)   [anchor chính]
//   E2 — Depth consistency: ~400 depth points phải  [lợi thế so với solvePnP]
//        nằm trên mặt phẳng marker theo pose T,
//        weighted theo depth confidence (Ý tưởng 6)
//   E3 — Square rigidity: 4 corners trong camera    [Ý tưởng 2]
//        frame phải tạo hình vuông cạnh L=0.06m
//   E4 — Multi-marker rigidity khi có >=2 marker    [lợi thế lớn nhất]
//        (rigid body constraint, giảm 12DOF → 6DOF)
//   E5 — Temporal regularization trên SO(3)         [thay thế EMA scalar]
//        (geodesic distance với pose frame trước)
//
// Outlier depth được xử lý bằng Huber loss (thay RANSAC).
// Optimizer: Gauss-Newton với line search, tự implement, không cần Ceres/g2o.
// Initial guess: lấy từ plane normal + center_3d của pipeline cũ (warm start).
//
// Kết quả: R* ∈ SO(3) đảm bảo → Roll/Pitch/Yaw chỉ là Euler decompose từ R*,
// không còn cross-axis coupling vì cả 3 trục được tối ưu đồng thời.
//
// Dependencies: OpenCV (cv::Mat, cv::Point2f, cv::Point3f) — không thêm gì.
// Target: RK3588, ~4–8ms/marker, trong budget 80ms/frame.
// ============================================================================
 
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
 
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
 
namespace se3opt {
 
// ----------------------------------------------------------------------------
// Intrinsics camera — tương tự dmsg.intr trong code gốc
// ----------------------------------------------------------------------------
struct CameraIntrinsics {
    float fx  = 0.f;
    float fy  = 0.f;
    float ppx = 0.f;
    float ppy = 0.f;
};
 
// ----------------------------------------------------------------------------
// Một điểm depth đã được deproject sang 3D, kèm confidence weight.
// weight = 1/σ²_depth, tính từ variance depth trong vùng lân cận pixel.
// ----------------------------------------------------------------------------
struct DepthPoint3D {
    cv::Point3f p;       // tọa độ 3D camera frame
    float       weight;  // confidence [0,1]; 1.0 = depth tốt nhất
};
 
// ----------------------------------------------------------------------------
// Input cho optimizer — dữ liệu của MỘT marker trong một frame
// ----------------------------------------------------------------------------
struct MarkerObservation {
    int                        id        = -1;
    std::array<cv::Point2f, 4> corners2d;   // pixel corners từ ArUco detect
    std::vector<DepthPoint3D>  depth_pts;   // điểm 3D từ depth frame (tối đa 400)
    bool                       valid     = false;

    // [FIX][E2'] Mat phang da do robust boi RANSAC+OLS (getPlaneDepthRobust),
    // dung de ghim TINH TIEN t (khong dung r) ve dung mat phang nay — xem
    // computeTranslationPlaneAnchorResidual() va Config::w_trans_plane.
    // E2' KHONG tach rieng t_z hay nx/ny, ma ghim ca 3 chieu cua t cung luc
    // theo dung khoang-cach-diem-toi-mat-phang hinh hoc that (anchor_normal
    // . t + d_anchor), va TUYET DOI KHONG dung den r0,r1,r2 (Jacobian phan
    // rotation = 0), nen ve mat dai so khong the truc tiep keo lech
    // Roll/Pitch/Yaw dang tinh dung tu E1+E3.
    //   anchor_normal: normal don vi (a,b,-1)/|...|, huong ve camera —
    //                  chinh la result.normal cua getPlaneDepthRobust().
    //   anchor_center: diem tam marker (X,Y,Z) tren mat phang do (camera
    //                  frame, CHUA offset hieu chuan) — dung de tinh
    //                  d_anchor = -(anchor_normal . anchor_center).
    cv::Point3f                anchor_normal{0.f, 0.f, -1.f};
    cv::Point3f                anchor_center{0.f, 0.f, 0.f};
    bool                       has_anchor_normal = false;
};
 
// ----------------------------------------------------------------------------
// Kết quả optimizer cho MỘT marker (hoặc rack frame nếu multi-marker)
// ----------------------------------------------------------------------------
struct SE3PoseResult {
    // Rotation matrix R ∈ SO(3), 3x3 float
    cv::Mat R;           // CV_32F, 3×3, R*Rᵀ=I, det=1
    // Translation vector t, camera frame (đơn vị: meter)
    cv::Point3f t;
 
    // Euler angles suy ra từ R (convention: ZYX = roll-pitch-yaw)
    double roll_deg  = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg   = 0.0;
 
    // Chất lượng
    double final_cost    = 0.0;  // E(ξ*) sau optimize
    double reproj_rmse_px = 0.0; // RMSE reprojection error (pixels)
    double depth_rmse_mm  = 0.0; // RMSE depth residual (mm)
    int    iterations     = 0;
    bool   converged      = false;
    bool   valid          = false;
};
 
// ----------------------------------------------------------------------------
// SE(3) Optimizer — main class
// ----------------------------------------------------------------------------
class SE3PoseOptimizer {
public:
    // ---- Tuning parameters ----
    struct Config {
        // Kích thước marker vật lý (cạnh hình vuông), đơn vị meter
        float marker_size_m = 0.06f;  // 6cm
 
        // Weights cho từng term trong cost function
        // Đơn vị: reproj tính bằng pixel², depth tính bằng m²
        // w_depth được auto-scale theo (σ_reproj/σ_depth)² ~ (1px/0.003m)²
        double w_reproj  = 1.0;   // E1: reprojection
        double w_depth   = 50.0;  // E2: depth consistency (~(1/0.003m)² normalized)

        // [FIX][E2'] Weight cho translation-plane-anchor. Ghim t (3 chieu)
        // ve dung mat phang RANSAC+OLS da do (obs.anchor_normal/anchor_center),
        // KHONG dung r (xem MarkerObservation::anchor_normal va
        // computeTranslationPlaneAnchorResidual). Muc dich: giam depth_rmse
        // (dang lech ~5-8x so voi RANSAC+OLS goc do bas-relief nhe khi
        // khong co anchor) MA KHONG dung cham den Roll/Pitch/Yaw dang tinh
        // dung tu E1+E3. Bat dau o muc VUA PHAI (thap hon w_square nhieu)
        // de tranh lan at E1 — tang dan va theo doi goc moi lan tang, dung
        // ngay khi thay goc bat dau troi so voi baseline (xem ghi chu hoi
        // thoai / README neu co). Neu obs.has_anchor_normal=false, term nay
        // tu dong bi bo qua (residual/Jacobian = 0), khong anh huong optimize.
        double w_trans_plane = 15.0; // E2': pin t (3D) ve mat phang RANSAC+OLS

        double w_square  = 200.0; // E3: square rigidity (hard constraint)
        double w_rigid   = 100.0; // E4: multi-marker rigidity
        double w_temporal = 0.15; // E5: temporal regularization (nhỏ để không lag)
 
        // Huber loss delta cho depth term (meter)
        // [FIX] 5mm qua chat so voi nhieu depth thuc te (~7-8mm quan sat duoc
        // qua log: SE3 depth_rmse ~7.5mm cho ID1, ~3.3mm cho ID2). Voi delta
        // qua nho, PHAN LON diem depth bi Huber downweight gan het (vi
        // |raw_r| > delta ngay ca khi la inlier that), mat tin hieu E2 ->
        // E1 (reprojection) lan at, ma voi marker gan fronto-parallel thi
        // 4 goc 2D gan nhu luon fit gan-hoan-hao voi rat nhieu pose khac
        // nhau (bas-relief ambiguity) -> pose bi keo lech khoi mat phang
        // khop tot nhat, depth_rmse SAU toi uu con TE HON ca RANSAC+OLS goc.
        // Tang len ~10mm de Huber chi thuc su downweight outlier ro rang,
        // giu duoc tin hieu dinh huong tu da so diem depth binh thuong.
        float huber_delta_m = 0.010f; // 10mm
 
        // Gauss-Newton / LM params
        int    max_iter        = 20;
        double cost_tol        = 1e-6;  // dừng khi Δcost < tol
        double step_tol        = 1e-6;  // dừng khi ‖Δξ‖ < tol ([FIX] 1e-7 qua chat, mixed rad+m units, khien LM luon dung het max_iter ma khong bao gio "chinh thuc" hoi tu du van cai thien tot)
        double lm_lambda_init  = 1e-3;  // Levenberg-Marquardt damping init
        double lm_lambda_max   = 1e3;
        double lm_lambda_min   = 1e-7;
 
        // Fallback: nếu optimizer không hội tụ hoặc cost quá cao,
        // dùng kết quả từ pipeline cũ (warm start) thay vì output rác
        double max_acceptable_reproj_px = 3.0; // pixel — nếu vượt → fallback
    };
 
    // [FIX] GCC khong cho dung "Config{}" (NSDMI cua struct long ben trong)
    // lam default argument ngay trong chinh class dang khai bao no ("default
    // member initializer required before the end of its enclosing class").
    // Tach thanh 2 constructor de tranh danh gia Config{} tai diem khai bao;
    // "= default" chi duoc sinh code khi thuc su duoc dung (odr-use), luc do
    // Config da la kieu hoan chinh.
    SE3PoseOptimizer() = default;
    explicit SE3PoseOptimizer(const Config& cfg) : _cfg(cfg) {}

    // [FIX][E2'-CHECK] Getter read-only cho Config — dung de log/debug xac
    // nhan gia tri weight thuc te dang active tai runtime (vd w_trans_plane),
    // tranh phai hardcode lai gia tri trong code goi ben ngoai (de bi lech
    // neu sau nay doi weight trong loadParams()/Config ma quen sua log).
    const Config& getConfig() const { return _cfg; }
 
    // ---- API chính ----
 
    // Optimize pose cho MỘT marker.
    // init_R, init_t: initial guess từ pipeline cũ (plane normal + center_3d).
    // Nếu !init_R.empty(), dùng làm warm start.
    SE3PoseResult optimizeSingle(
        const MarkerObservation&  obs,
        const CameraIntrinsics&   intr,
        const cv::Mat&            init_R,    // 3×3 CV_32F hoặc empty
        const cv::Point3f&        init_t,
        const cv::Mat&            prev_R,    // frame trước, cho E5; empty = bỏ qua
        const cv::Point3f&        prev_t);
 
    // Optimize pose chung (rack frame) cho NHIỀU marker đồng thời.
    // known_offsets_m[i]: vị trí của marker i trong rack frame (đã đo vật lý).
    // Optimize 1 SE(3) duy nhất thay vì N SE(3) độc lập.
    SE3PoseResult optimizeMulti(
        const std::vector<MarkerObservation>&  obs_list,
        const CameraIntrinsics&                intr,
        const std::vector<cv::Point3f>&        known_offsets_m,
        const cv::Mat&                         init_R,
        const cv::Point3f&                     init_t,
        const cv::Mat&                         prev_R,
        const cv::Point3f&                     prev_t);
 
    // Helper: build initial guess từ plane normal + center_3d (pipeline cũ)
    // Dùng để tạo init_R, init_t trước khi gọi optimize*
    static bool buildInitialGuess(
        const cv::Point3f& plane_normal,   // unit normal, camera frame
        const cv::Point3f& center_3d,      // center của marker, camera frame
        const std::array<cv::Point2f, 4>& corners2d,
        const CameraIntrinsics& intr,
        float marker_size_m,
        cv::Mat& out_R,
        cv::Point3f& out_t);
 
    // Helper: extract Roll/Pitch/Yaw từ R (ZYX convention)
    // pitch = atan2(R[2][0], sqrt(R[0][0]²+R[1][0]²)) - KHÔNG dùng atan2(ny,-nz)
    static void rotMatToRPY(const cv::Mat& R,
                             double& roll_deg,
                             double& pitch_deg,
                             double& yaw_deg);
 
private:
    Config _cfg;
 
    // ---- Rodrigues parameterization ----
    // ξ = [r0, r1, r2, t0, t1, t2], r = rotation vector (Rodrigues)
    // R = exp([r]×) theo Rodrigues formula
    static cv::Mat rodriguesToR(const double r[3]);
    static void    rToRodrigues(const cv::Mat& R, double r[3]);
 
    // dR/dr: Jacobian của R(r) theo r, dùng cho Gauss-Newton
    // Trả về 3 ma trận 3×3 (partial derivatives)
    static void rodriguesToRJacobian(const double r[3],
                                      cv::Mat dRdr[3]);
 
    // ---- Cost terms ----
 
    // E1: reprojection residuals, 2×4 = 8 scalars
    // res[2k], res[2k+1] = (u_proj - u_obs), (v_proj - v_obs) cho corner k
    void computeReprojResiduals(
        const double xi[6],   // [r0..2, t0..2]
        const std::array<cv::Point2f, 4>& corners2d,
        const CameraIntrinsics& intr,
        double res[8],
        double J[8][6]) const; // Jacobian nếu không null
 
    // E2: depth residuals với Huber weighting
    // Trả về vector residual (1 per depth point), Jacobian (Nx6)
    void computeDepthResiduals(
        const double xi[6],
        const std::vector<DepthPoint3D>& pts,
        const CameraIntrinsics& intr,
        std::vector<double>& res,
        std::vector<std::array<double,6>>& J) const;

    // [FIX][E2'] Translation-plane-anchor residual: ghim tinh tien t (xi[3..5])
    // ve dung mat phang da do boi RANSAC+OLS (anchor_normal, anchor_center).
    // 1 scalar residual = anchor_normal . t + d_anchor (khoang cach t den
    // mat phang do, don vi met). Jacobian 1x6: J[0..2]=0 (KHONG phu thuoc
    // rotation r), J[3..5] = anchor_normal.x/y/z. Xem ghi chu day du tai
    // Config::w_trans_plane va MarkerObservation::anchor_normal.
    void computeTranslationPlaneAnchorResidual(
        const double xi[6],
        const cv::Point3f& anchor_normal,
        const cv::Point3f& anchor_center,
        double& res,
        double J[6]) const;
 
    // E3: square rigidity residuals, 5 scalars
    // [‖P0P1‖-L, ‖P1P2‖-L, ‖P2P3‖-L, ‖P3P0‖-L, P0P1·P1P2/L²]
    void computeSquareResiduals(
        const double xi[6],
        double res[5],
        double J[5][6]) const;
 
    // E5: temporal residuals, 6 scalars [r_delta(0..2), t_delta(0..2)]
    void computeTemporalResiduals(
        const double xi[6],
        const cv::Mat& prev_R,
        const cv::Point3f& prev_t,
        double res[6],
        double J[6][6]) const;
 
    // ---- Gauss-Newton core ----
    // Accumulate JᵀJ và Jᵀr từ một block residual
    static void accumJtJJtr(
        const double* res, const double* J_flat,
        int n_res, int n_param,
        double JtJ[6][6], double Jtr[6]);
 
    // Solve (JᵀJ + λI)Δξ = -Jᵀr (LM step)
    static bool solveLM(double JtJ[6][6], double Jtr[6],
                         double lambda, double dxi[6]);
 
    // Project một điểm 3D (camera frame) lên ảnh
    static void project(const cv::Point3f& p3d, const CameraIntrinsics& intr,
                         double& u, double& v);
 
    // Transform một điểm từ marker frame sang camera frame qua (R,t)
    // marker frame: center ở origin, corners tại ±L/2
    cv::Point3f transformPoint(const double xi[6], const cv::Point3f& p_marker) const;
 
    // Corners của marker trong marker frame (đã biết từ L)
    // Thứ tự ArUco chuẩn: P0=TL, P1=TR, P2=BR, P3=BL
    // (0,0) = center, X right, Y down, Z toward camera
    std::array<cv::Point3f, 4> markerCorners3D() const;
 
    // Huber weight rho'(r²) = min(1, delta/|r|) cho scalar residual r
    double huberWeight(double r, double delta) const;
 
    // [FIX] Truoc la 1 free function ben ngoai class nhung lai goi thang
    // cac ham private (computeReprojResiduals/...) qua 1 instance opt ->
    // khong compile duoc (private within this context). Chuyen thanh member
    // function private cua chinh SE3PoseOptimizer de dung "this" tu nhien,
    // khong can truyen instance/Config rieng nua.
    double computeTotalCost(
        const double xi[6],
        const MarkerObservation& obs,
        const CameraIntrinsics& intr,
        const cv::Mat& prev_R,
        const cv::Point3f& prev_t) const;
};
 
// ----------------------------------------------------------------------------
// Helper: build MarkerObservation từ dữ liệu depth frame
// (dùng trong tryBuildDetection() thay cho getPlaneDepthRobust)
// ----------------------------------------------------------------------------
struct DepthFrameView {
    const uint16_t* z16          = nullptr;
    int             width        = 0;
    int             height       = 0;
    int             stride_u16   = 0;
    float           depth_scale  = 0.f;
};
 
// Sample depth points trong polygon của marker, compute per-point weight
// từ local depth variance (thực hiện Ý tưởng 6).
// step=3 để giữ ngân sách CPU như cũ, max_points=400.
MarkerObservation buildMarkerObservation(
    int                               marker_id,
    const std::array<cv::Point2f, 4>& corners2d,
    const DepthFrameView&             depth,
    const CameraIntrinsics&           intr,
    int                               step      = 3,
    int                               max_points = 400,
    int                               var_half_win = 2);  // cửa sổ tính variance
 
} // namespace se3opt
// ============================================================================
// [SE3] Het phan gop tu se3_pose_optimizer.h
// ============================================================================
 
// ============================================================================
// [PLANE DEPTH + NORMAL] Ket qua fit mat phang tren polygon marker: Z tai
// tam marker (refined bang OLS) + plane normal (camera frame). Normal dung
// lam nguon Roll/Pitch, tuong tu main11.cpp — KHONG dung solvePnP.
// ============================================================================
struct ArucoPlaneDepthResult {
    float       depth_m    = 0.0f;
    float       rmse_mm    = 0.0f;
    int         inliers    = 0;
    bool        valid      = false;
    cv::Point3f normal{0.0f, 0.0f, 1.0f};
    bool        has_normal = false;
};
 
// ============================================================================
// [ORIENTATION] Goc Roll/Pitch/Yaw cua giá đỡ so với camera.
//  - Roll, Yaw : uu tien tinh hinh hoc tu vector noi tam 2 marker (khi co
//    >=2 marker hop le trong frame). Yaw fallback sang plane-normal khi chi
//    thay 1 marker.
//  - Pitch     : luon tinh tu plane normal trung binh (weighted theo RMSE),
//    vi khong the tach duoc tu vector noi 2 tam marker.
// Xem main11.cpp phan compute_orientation_from_markers() de biet chi tiet
// ly do thiet ke / cong thuc.
// ============================================================================
struct ArucoOrientationState {
    double roll_deg  = 0.0;   // quay quanh Z (truc nhin)
    double pitch_deg = 0.0;   // quay quanh X (ngan/cup)
    double yaw_deg   = 0.0;   // quay quanh Y (lac trai/phai)
    bool   roll_valid  = false;
    bool   pitch_valid = false;
    bool   yaw_valid   = false;
 
    // [RACK FRAME] He truc local cua gia hang, dung truc tiep tu normal +
    // duong noi 2 tam marker (Gram-Schmidt). Chi valid khi co >=2 marker.
    bool        rack_frame_valid = false;
    cv::Point3f rack_origin{0.f, 0.f, 0.f};    // trung diem 2 tam marker, camera frame (KHONG offset)
    cv::Point3f rack_x_axis{1.f, 0.f, 0.f};    // doc gia, trai -> phai
    cv::Point3f rack_y_axis{0.f, 1.f, 0.f};    // doc gia, tren -> duoi
    cv::Point3f rack_z_axis{0.f, 0.f, -1.f};   // vuong goc mat gia, huong ve camera
};
 
class ArucoDetectorNew : public Plugin
{
    ArucoPlaneDepthResult getPlaneDepthRobust(const tmv::DepthFrame& dmsg,
                              const std::vector<cv::Point2f>& corners,
                              int marker_id) const;
 
    // [PITCH FIX] Fit 1 mat phang DUY NHAT tren toan bo diem 3D gop tu tat
    // ca marker truyen vao (thay vi fit rieng tung marker roi gop normal).
    // Baseline khong gian tang tu vai cm (1 marker) len ~300mm (khoang cach
    // 2 marker) -> giam manh sai so uoc luong do nghieng (a,b), vi sai so
    // do nghieng ti le nghich voi do trai rong khong gian cua tap diem fit.
    // Chi goi khi co >=2 marker hop le trong frame. Ket qua chi dung cho
    // Pitch (va rack z-axis) — Z rieng tung marker van lay tu
    // getPlaneDepthRobust() ben tren, khong bi anh huong.
    ArucoPlaneDepthResult getJointPlaneNormal(const tmv::DepthFrame& dmsg,
                              const std::vector<std::vector<cv::Point2f>>& cornersList) const;
 
public:
    ArucoDetectorNew(const std::string& name);
    ~ArucoDetectorNew();
 
    void run() override;
 
    // [ORIENTATION] Truy cap ket qua Roll/Pitch/Yaw cua frame gan nhat.
    const ArucoOrientationState& getOrientation() const { return _orientState; }
 
private:
    void loadParams();
    bool tryBuildDetection(tmv::MarkerDetections& out);
 
    cv::Mat applySpatialFilter(const cv::Mat& gray) const;
    cv::Mat applyTemporalFilter(const cv::Mat& gray);
 
    static inline int clampi(int v, int lo, int hi) {
        return (v < lo) ? lo : (v > hi) ? hi : v;
    }
 
    float depthMedianAt(const tmv::DepthFrame& dmsg, int u, int v, int half_win) const;
 
    bool deprojectXYZ(float& X, float& Y, float& Z,
                      float u, float v,
                      float depth_m,
                      float fx, float fy, float ppx, float ppy) const;
 
    // ------------------------------------------------------------------
    // [ORIENTATION] Sample toi thieu can de tinh Roll/Pitch/Yaw cho 1
    // marker trong 1 frame. center_3d la toa do camera-frame CHUA offset
    // (khong dung offsetX/Y/Z hieu chuan cua plugin cho tinh toan goc).
    // ------------------------------------------------------------------
    struct MarkerPoseSample {
        int         id = -1;
        cv::Point2f center_px{0.f, 0.f};
        cv::Point3f center_3d{0.f, 0.f, 0.f};
        cv::Point3f normal{0.f, 0.f, 1.f};
        bool        has_normal = false;
        float       rmse_mm = 0.0f;
    };
 
    struct ScalarFilterState {
        bool   valid = false;
        double value = 0.0;
    };
 
    static void applyScalarFilter(ScalarFilterState& s, double raw_value,
                                  double alpha, bool is_angle_deg);
 
    // Cap nhat _orientState + cac EMA filter tu danh sach marker sample hop
    // le cua frame hien tai. Neu < 1 marker co normal, giu nguyen gia tri
    // truoc do va dat *_valid = false.
    //
    // [PITCH FIX] has_joint_normal/joint_normal/joint_rmse_mm: ket qua tu
    // getJointPlaneNormal() (baseline ~300mm). Khi has_joint_normal=true,
    // Pitch (va rack z-axis) uu tien dung normal nay thay vi weighted
    // average normal tung marker (baseline chi vai cm, sai so lon hon).
    void computeOrientationFromMarkers(const std::vector<MarkerPoseSample>& samples,
                                       bool has_joint_normal = false,
                                       const cv::Point3f& joint_normal = cv::Point3f(0.f, 0.f, -1.f),
                                       float joint_rmse_mm = 0.0f);
 
    // ------------------------------------------------------------------
    // [SE3] Ap dung ket qua optimizeMulti() (1 SE(3) rigid cho toan bo
    // rack, E4) truc tiep vao _orientState — KHONG can atan2 tu vector noi
    // 2 tam marker nua vi R* da chua san 3 truc truc giao (Roll/Pitch/Yaw
    // duoc doc thang tu R* qua rotMatToRPY, dung convention Z=roll,
    // X=pitch, Y=yaw giong het ArucoOrientationState).
    // ------------------------------------------------------------------
    void applySE3RackOrientation(const se3opt::SE3PoseResult& rack);
 
private:
    std::atomic<bool> _running{false};
 
    int _dictId = static_cast<int>(cv::aruco::DICT_6X6_100);
    cv::Ptr<cv::aruco::Dictionary>         _dict;
    cv::Ptr<cv::aruco::DetectorParameters> _params;
 
    uint64_t _lastProcKey   = 0;
    int      _maxSyncDiffMs = 50;
    uint64_t _maxSyncDiffNs = 50ULL * 1000ULL * 1000ULL;
    int      _loopSleepMs   = 100;
 
    int  _roiHalfWin   = 3;
    bool _enableXYZ    = true;
    bool _enableStdout = true;
 
    bool   _enableSpatialFilter = true;
    int    _spatialGaussKernel  = 3;
    double _spatialGaussSigma   = 0.8;
    bool   _enableCLAHE         = false;
    double _claheClipLimit      = 2.0;
    int    _claheTileSize       = 8;
 
    bool   _enableTemporalFilter = true;
    double _temporalEmaAlpha     = 0.4;
 
    cv::Mat _grayEmaAccum;
    bool    _grayEmaInit = false;
 
    cv::Ptr<cv::CLAHE> _clahe;
    // Calibration offset (RIENG cua plugin nay — KHONG lien quan
    // apply_depth_correction()/transform_to_mechanical_front() cua main11).
    float _offsetX = -0.0175f;
    float _offsetY =  0.0f;
    float _offsetZ =  0.0f;
 
    // ------------------------------------------------------------------
    // [ORIENTATION] Normal accumulator per marker ID — weighted average
    // theo 1/rmse qua buffer 12 frame, reject frame co RMSE > 5mm. Giup
    // giam noise cua normal 1-marker (~±5-13° raw) xuong muc chap nhan
    // duoc (~±1-2°) truoc khi dung cho Pitch/Yaw-fallback.
    // ------------------------------------------------------------------
    struct NormalAccumulator {
        struct Entry {
            cv::Point3f normal;
            float       rmse_mm;
        };
        std::deque<Entry> buffer;
        static constexpr int   kMaxSize      = 12;
        static constexpr float kRmseRejectMm = 5.0f;
 
        void push(const cv::Point3f& n, float rmse_mm) {
            if (rmse_mm > kRmseRejectMm) return;
            buffer.push_back({n, rmse_mm});
            if (static_cast<int>(buffer.size()) > kMaxSize) buffer.pop_front();
        }
 
        cv::Point3f averaged() const {
            if (buffer.empty()) return {0.f, 0.f, -1.f};
            cv::Point3f sum(0.f, 0.f, 0.f);
            const float eps = 0.1f;
            for (const auto& e : buffer) {
                float w = 1.0f / (e.rmse_mm + eps);
                sum.x += w * e.normal.x;
                sum.y += w * e.normal.y;
                sum.z += w * e.normal.z;
            }
            float len = std::sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
            if (len < 1e-9f) return buffer.back().normal;
            return sum * (1.0f / len);
        }
 
        bool empty() const { return buffer.empty(); }
    };
    std::map<int, NormalAccumulator> _normalAccum;
 
    // ------------------------------------------------------------------
    // [SE3] Optimizer SE(3) chinh + trang thai warm-start (E5 temporal).
    // Thay vi EMA scalar tren roll/pitch/yaw sau khi tinh (nhu cu), pose
    // (R,t) cua frame truoc duoc dua thang vao cost function lam temporal
    // regularization (geodesic SLERP tren SO(3)), khop hon EMA goc-Euler.
    // ------------------------------------------------------------------
    se3opt::SE3PoseOptimizer _se3Opt;
    float                    _markerSizeM = 0.06f;  // do vat ly, dat trong loadParams()
 
    struct PrevPoseSE3 {
        cv::Mat     R;                       // CV_32F 3x3, rong = chua co warm-start
        cv::Point3f t{0.f, 0.f, 0.f};
        bool        valid = false;
    };
    std::map<int, PrevPoseSE3> _prevPoseByMarker;   // warm-start per-marker (optimizeSingle)
    PrevPoseSE3                _prevRackPose;        // warm-start rack pose (optimizeMulti, E4)
 
    // [SE3][E4] Vi tri tam moi marker trong rack frame, do vat ly 1 lan
    // (khong phai tinh dong moi frame). Neu rong -> khong du dieu kien
    // chay optimizeMulti(), tu dong fallback ve tung marker doc lap +
    // computeOrientationFromMarkers() nhu pipeline cu (nhung voi input da
    // duoc SE3 lam sach o buoc per-marker).
    std::map<int, cv::Point3f> _knownRackOffsetsM;
 
    // [PITCH FIX v2] Joint-plane normal (fit gop 2 marker) hien tai duoc tinh
    // lai hoan toan moi frame, khong co tich luy theo thoi gian nhu normal
    // tung marker o tren. Vi baseline gop chi tang theo truc X (2 marker dat
    // ngang hang), khong tang theo truc Y (baseline cua Pitch), ny van con
    // nhieu ~ tuong duong muc 1-marker truoc day (xem log thuc te). Buffer
    // nay giam bot phan nhieu do RANSAC ngau nhien (chi 30 vong/frame) +
    // nhieu depth theo thoi gian — khong giai quyet duoc gioi han hinh hoc
    // baseline-Y, nhung giam duoc variance do thuc te.
    NormalAccumulator _jointNormalAccum;
 
    // [ORIENTATION] EMA filter state + ket qua Roll/Pitch/Yaw cua frame gan nhat
    ScalarFilterState     _rollFilter;
    ScalarFilterState     _pitchFilter;
    ScalarFilterState     _yawFilter;
    ArucoOrientationState _orientState;
    const double          _orientFilterAlpha  = 0.4;
    const bool            _enableOrientFilter = true;
};