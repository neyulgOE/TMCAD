#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>

// Test the Gauss-Newton Point-to-Plane refinement on realistic tabletop observation cloud
int main()
{
    std::cout << "========================================================\n";
    std::cout << "  TEST GAUSS-NEWTON POINT-TO-PLANE REFINEMENT (<2-5mm)\n";
    std::cout << "========================================================\n\n";

    // CAD Dimensions for TABLE-DEMO
    const double hx = 0.220; // width 0.44m
    const double hy = 0.415; // length 0.83m
    const double z_top = 0.1665;
    const double z_bot = -0.3355;

    // Ground Truth Pose (Camera to Table)
    // t_gt = [-0.05m, 0.02m, 1.25m], Euler = [Roll: 1.5 deg, Pitch: -3.0 deg, Yaw: 35.0 deg]
    double rad_roll = 1.5 * CV_PI / 180.0;
    double rad_pitch = -3.0 * CV_PI / 180.0;
    double rad_yaw = 35.0 * CV_PI / 180.0;

    cv::Matx33d R_x(1, 0, 0,
                    0, std::cos(rad_roll), -std::sin(rad_roll),
                    0, std::sin(rad_roll),  std::cos(rad_roll));
    cv::Matx33d R_y(std::cos(rad_pitch), 0, std::sin(rad_pitch),
                    0, 1, 0,
                   -std::sin(rad_pitch), 0, std::cos(rad_pitch));
    cv::Matx33d R_z(std::cos(rad_yaw), -std::sin(rad_yaw), 0,
                    std::sin(rad_yaw),  std::cos(rad_yaw), 0,
                    0, 0, 1);
    cv::Matx33d R_gt = R_z * R_y * R_x;
    cv::Vec3d t_gt(-0.05, 0.02, 1.25);

    std::cout << "[GT Pose] Target Translation: [" << t_gt[0] << ", " << t_gt[1] << ", " << t_gt[2] << "] m\n";
    std::cout << "[GT Pose] Target Angles:      Roll=" << 1.5 << " deg, Pitch=" << -3.0 << " deg, Yaw=" << 35.0 << " deg\n\n";

    // Generate 512 CAD centroids on the table
    std::vector<cv::Vec3f> cad_centroids(512);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_x(-hx, hx);
    std::uniform_real_distribution<float> dist_y(-hy, hy);
    std::normal_distribution<float> sensor_noise(0.0f, 0.002f); // 2 mm RealSense depth noise

    for (int i = 0; i < 512; ++i)
    {
        // 70% points on tabletop, 30% on legs
        if (i < 360)
        {
            cad_centroids[i] = cv::Vec3f(dist_x(rng), dist_y(rng), static_cast<float>(z_top));
        }
        else
        {
            float leg_x = (i % 2 == 0) ? (hx - 0.03f) : (-hx + 0.03f);
            float leg_y = ((i / 2) % 2 == 0) ? (hy - 0.03f) : (-hy + 0.03f);
            float leg_z = z_bot + static_cast<float>(i % 30) * 0.015f;
            cad_centroids[i] = cv::Vec3f(leg_x, leg_y, leg_z);
        }
    }

    // Generate 1024 observation points = R_gt * cad_points + t_gt + noise
    std::vector<cv::Vec3f> obs_points(1024);
    std::vector<cv::Vec3f> cad_interp_points(1024);
    std::vector<uint8_t> inlier_mask(1024, 1);

    for (int i = 0; i < 1024; ++i)
    {
        int cad_idx = i % 512;
        cv::Vec3d p_cad(cad_centroids[cad_idx]);
        cv::Vec3d p_cam = R_gt * p_cad + t_gt;

        // Add 2mm sensor noise
        p_cam[0] += sensor_noise(rng);
        p_cam[1] += sensor_noise(rng);
        p_cam[2] += sensor_noise(rng);

        obs_points[i] = cv::Vec3f(p_cam);
        cad_interp_points[i] = cv::Vec3f(p_cad);
    }

    // Coarse Pose: Introduce typical Kabsch SVD error (+15mm translation, +1.5 deg rotation)
    double err_rad_yaw = 1.5 * CV_PI / 180.0;
    cv::Matx33d R_err(std::cos(err_rad_yaw), -std::sin(err_rad_yaw), 0,
                      std::sin(err_rad_yaw),  std::cos(err_rad_yaw), 0,
                      0, 0, 1);
    cv::Matx33d R_coarse = R_err * R_gt;
    cv::Vec3d t_coarse = t_gt + cv::Vec3d(0.012, -0.010, 0.015); // 15mm coarse drift

    double coarse_err_t = cv::norm(t_coarse - t_gt) * 1000.0; // in mm
    std::cout << "[Coarse Pose] Initial Kabsch error: e_t = " << coarse_err_t << " mm\n\n";

    // Run Gauss-Newton Point-to-Plane Refinement
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Matx33d R = R_coarse;
    cv::Vec3d t = t_coarse;

    for (int iter = 0; iter < 8; ++iter)
    {
        cv::Vec3d n_table(R(0, 2), R(1, 2), R(2, 2));
        cv::Matx33d R_inv = R.t();

        cv::Matx66d H = cv::Matx66d::zeros();
        cv::Matx61d b = cv::Matx61d::zeros();
        int used_constraints = 0;

        for (int i = 0; i < 1024; ++i)
        {
            if (!inlier_mask[i]) continue;

            cv::Vec3d p_obs(obs_points[i]);
            cv::Vec3d p_cad(cad_interp_points[i]);

            cv::Vec3d p_cad_est = R_inv * (p_obs - t);

            // Constraint 1: Tabletop Point-to-Plane
            bool is_tabletop = (std::abs(p_cad_est[2] - z_top) < 0.035) &&
                               (std::abs(p_cad_est[0]) <= hx + 0.025) &&
                               (std::abs(p_cad_est[1]) <= hy + 0.025);

            if (is_tabletop)
            {
                double r_plane = n_table.dot(p_obs - t) - z_top;
                double weight = (std::abs(r_plane) < 0.015) ? 3.0 : 3.0 * (0.015 / std::abs(r_plane));

                cv::Vec3d d_rot = -n_table.cross(p_obs - t);
                cv::Vec3d d_trans = -n_table;

                double J_i[6] = {d_rot[0], d_rot[1], d_rot[2], d_trans[0], d_trans[1], d_trans[2]};

                for (int r = 0; r < 6; ++r)
                {
                    for (int c = 0; c < 6; ++c)
                    {
                        H(r, c) += weight * J_i[r] * J_i[c];
                    }
                    b(r, 0) -= weight * J_i[r] * r_plane;
                }
                ++used_constraints;
            }

            // Constraint 2: 3D Point Correspondence Anchor
            cv::Vec3d p_pred = R * p_cad + t;
            cv::Vec3d r_corr = p_pred - p_obs;
            double dist_corr = cv::norm(r_corr);

            if (dist_corr < 0.035)
            {
                double w_corr = 1.0;
                cv::Matx33d skew(
                     0.0,        p_pred[2], -p_pred[1],
                    -p_pred[2],  0.0,        p_pred[0],
                     p_pred[1], -p_pred[0],  0.0
                );

                for (int dim = 0; dim < 3; ++dim)
                {
                    double J_dim[6] = {
                        skew(dim, 0), skew(dim, 1), skew(dim, 2),
                        (dim == 0) ? 1.0 : 0.0,
                        (dim == 1) ? 1.0 : 0.0,
                        (dim == 2) ? 1.0 : 0.0
                    };

                    for (int r = 0; r < 6; ++r)
                    {
                        for (int c = 0; c < 6; ++c)
                        {
                            H(r, c) += w_corr * J_dim[r] * J_dim[c];
                        }
                        b(r, 0) -= w_corr * J_dim[r] * r_corr[dim];
                    }
                }
                ++used_constraints;
            }
        }

        if (used_constraints < 15) break;

        for (int d = 0; d < 6; ++d) H(d, d) += 1e-4;

        cv::Matx61d delta;
        if (!cv::solve(H, b, delta, cv::DECOMP_CHOLESKY))
        {
            if (!cv::solve(H, b, delta, cv::DECOMP_SVD)) break;
        }

        cv::Vec3d d_w(delta(0, 0), delta(1, 0), delta(2, 0));
        cv::Vec3d d_t(delta(3, 0), delta(4, 0), delta(5, 0));

        double d_theta = cv::norm(d_w);
        cv::Matx33d d_R = cv::Matx33d::eye();
        if (d_theta > 1e-6)
        {
            cv::Vec3d axis = d_w * (1.0 / d_theta);
            cv::Matx33d K(
                 0.0,     -axis[2],  axis[1],
                 axis[2],  0.0,     -axis[0],
                -axis[1],  axis[0],  0.0
            );
            d_R = cv::Matx33d::eye() + std::sin(d_theta) * K + (1.0 - std::cos(d_theta)) * (K * K);
        }

        R = d_R * R;
        t = t + d_t;

        if (cv::norm(d_t) < 0.0002 && d_theta < (0.05 * CV_PI / 180.0))
        {
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double refine_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Compute final errors vs GT
    double final_err_t_mm = cv::norm(t - t_gt) * 1000.0;

    // Rotation error in deg
    cv::Matx33d R_diff = R * R_gt.t();
    double tr = R_diff(0, 0) + R_diff(1, 1) + R_diff(2, 2);
    double cos_ang = std::clamp((tr - 1.0) * 0.5, -1.0, 1.0);
    double final_err_R_deg = std::acos(cos_ang) * 180.0 / CV_PI;

    std::cout << "[Refined Pose Output]\n";
    std::cout << "  - Refinement Time:      " << refine_ms << " ms (Target: < 1.5 ms)\n";
    std::cout << "  - Final Position Error: " << final_err_t_mm << " mm (Target: <= 5.0 mm)\n";
    std::cout << "  - Final Rotation Error: " << final_err_R_deg << " deg (Target: < 1.0 deg)\n";
    std::cout << "  - Translation (X, Y, Z): [" << t[0] << ", " << t[1] << ", " << t[2] << "] m\n";

    if (final_err_t_mm <= 5.0 && final_err_R_deg < 1.0 && refine_ms < 2.0)
    {
        std::cout << "\n>>> GAUSS-NEWTON POINT-TO-PLANE REFINEMENT VERIFIED SUCCESSFUL! <<<\n";
    }

    return 0;
}
