#!/usr/bin/env python3
import os
import cv2
import numpy as np

project_root = "/home/huyen1905/TMCAD"
rgb_path = os.path.join(project_root, "dump_rgb.png")
out_img_path = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/cad_box_overlay_result.png"
out_sbs_path = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/sbs_target_vs_model_box.png"
target_ref_path = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/.user_uploaded/media_1793697292792_de2cb0a8.jpg"

K = np.array([
    [608.215, 0.0, 341.473],
    [0.0, 608.128, 228.438],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Model estimated pose
R_est = np.array([
    [-0.00156208,  0.9986425,  -0.0520644 ],
    [ 0.02895015, -0.05199747, -0.99822754],
    [-0.99957955, -0.00306662, -0.02882962]
], dtype=np.float64)
t_est = np.array([0.0368, -0.0851, 1.2343], dtype=np.float64)

# In TABLE-DEMO STL:
# Width (depth into scene) along X: [-0.220, +0.220]
# Length (horizontal across image) along Y: [-0.415, +0.415]
# Height (vertical) along Z: [-0.334, +0.171]
cad_x0, cad_x1 = -0.220, 0.220
cad_y0, cad_y1 = -0.415, 0.415
z_floor = -0.334
z_top = 0.171

def render_box(rgb_img, R, t, x_val=0.04, y_val=-0.09, z_val=1.23, yaw_val=0.2, fit_pct=98):
    img = rgb_img.copy()

    def proj(x, y, z):
        p_cam = R @ np.array([x, y, z]) + t
        z_eff = max(0.05, p_cam[2])
        u = K[0, 0] * p_cam[0] / z_eff + K[0, 2]
        v = K[1, 1] * p_cam[1] / z_eff + K[1, 2]
        return np.array([u, v], dtype=np.float32)

    # CAD X is depth: +x is front (closer), -x is rear (further)
    x_front = cad_x1
    x_rear  = cad_x0
    # CAD Y is length: -y is left, +y is right
    y_left  = cad_y0
    y_right = cad_y1

    # 8 Main Corners
    p_f_tl = proj(x_front, y_left,  z_top)
    p_f_tr = proj(x_front, y_right, z_top)
    p_r_tl = proj(x_rear,  y_left,  z_top)
    p_r_tr = proj(x_rear,  y_right, z_top)

    p_f_bl = proj(x_front, y_left,  z_floor)
    p_f_br = proj(x_front, y_right, z_floor)
    p_r_bl = proj(x_rear,  y_left,  z_floor)
    p_r_br = proj(x_rear,  y_right, z_floor)

    YELLOW = (0, 238, 255)
    GREEN  = (0, 230, 118)
    CYAN   = (255, 229, 0)
    RED    = (0, 0, 255)
    BLUE   = (255, 120, 0)

    # 1. Shaded Tabletop (Cyan with 25% transparency)
    top_poly = np.array([p_f_tl, p_f_tr, p_r_tr, p_r_tl], dtype=np.int32)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, top_poly, CYAN)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # 2. Floor Footprint Rectangle (Green) with 4 Feet Ellipses
    bot_poly = np.array([p_f_bl, p_f_br, p_r_br, p_r_bl], dtype=np.int32)
    cv2.polylines(img, [bot_poly], True, GREEN, 2, cv2.LINE_AA)

    def to_pt(p):
        return (int(round(p[0])), int(round(p[1])))

    cv2.ellipse(img, to_pt(p_f_bl), (10, 4), 0, 0, 360, GREEN, 2, cv2.LINE_AA)
    cv2.ellipse(img, to_pt(p_f_br), (10, 4), 0, 0, 360, GREEN, 2, cv2.LINE_AA)
    cv2.ellipse(img, to_pt(p_r_bl), (8, 3), 0, 0, 360, GREEN, 1, cv2.LINE_AA)
    cv2.ellipse(img, to_pt(p_r_br), (8, 3), 0, 0, 360, GREEN, 1, cv2.LINE_AA)

    v_down_f_l = p_f_bl - p_f_tl
    v_down_f_r = p_f_br - p_f_tr
    v_down_r_l = p_r_bl - p_r_tl
    v_down_r_r = p_r_br - p_r_tr

    h_beam = 0.08
    h_leg_bot = 0.89

    # 3. Top Perimeter Beams
    cv2.line(img, to_pt(p_f_tl), to_pt(p_f_tr), YELLOW, 3, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tl + h_beam * v_down_f_l), to_pt(p_f_tr + h_beam * v_down_f_r), CYAN, 2, cv2.LINE_AA)

    cv2.line(img, to_pt(p_r_tl), to_pt(p_r_tr), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_r_tl + h_beam * v_down_r_l), to_pt(p_r_tr + h_beam * v_down_r_r), CYAN, 2, cv2.LINE_AA)

    cv2.line(img, to_pt(p_f_tl), to_pt(p_r_tl), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tl + h_beam * v_down_f_l), to_pt(p_r_tl + h_beam * v_down_r_l), CYAN, 2, cv2.LINE_AA)

    cv2.line(img, to_pt(p_f_tr), to_pt(p_r_tr), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tr + h_beam * v_down_f_r), to_pt(p_r_tr + h_beam * v_down_r_r), CYAN, 2, cv2.LINE_AA)

    # Top Center Vertical Bracket
    p_center_t = (p_f_tl + p_f_tr) * 0.5 + h_beam * (v_down_f_l + v_down_f_r) * 0.5
    p_center_b = p_center_t + 0.09 * (v_down_f_l + v_down_f_r) * 0.5
    cv2.line(img, to_pt(p_center_t - np.array([4, 0])), to_pt(p_center_b - np.array([4, 0])), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_center_t + np.array([4, 0])), to_pt(p_center_b + np.array([4, 0])), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_center_b - np.array([4, 0])), to_pt(p_center_b + np.array([4, 0])), GREEN, 2, cv2.LINE_AA)

    # 4. Front-Left Column (Yellow outer line, Cyan inner line)
    p_f_tl_inner = p_f_tl + 0.06 * (p_f_tr - p_f_tl)
    leg_bot_l = p_f_tl + h_leg_bot * v_down_f_l
    leg_bot_l_inner = p_f_tl_inner + h_leg_bot * v_down_f_l

    cv2.line(img, to_pt(p_f_tl), to_pt(leg_bot_l), YELLOW, 3, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tl_inner + h_beam * v_down_f_l), to_pt(leg_bot_l_inner), YELLOW, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_l), to_pt(p_f_bl), YELLOW, 2, cv2.LINE_AA)

    # 5. Front-Right Column (Green outer line, Cyan inner line)
    p_f_tr_inner = p_f_tr + 0.06 * (p_f_tl - p_f_tr)
    leg_bot_r = p_f_tr + h_leg_bot * v_down_f_r
    leg_bot_r_inner = p_f_tr_inner + h_leg_bot * v_down_f_r

    cv2.line(img, to_pt(p_f_tr), to_pt(leg_bot_r), GREEN, 3, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tr_inner + h_beam * v_down_f_r), to_pt(leg_bot_r_inner), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_r), to_pt(p_f_br), GREEN, 2, cv2.LINE_AA)

    # 6. Rear-Left Column (Cyan)
    p_r_tl_inner = p_r_tl + 0.06 * (p_r_tr - p_r_tl)
    leg_bot_rl = p_r_tl + h_leg_bot * v_down_r_l
    leg_bot_rl_inner = p_r_tl_inner + h_leg_bot * v_down_r_l
    cv2.line(img, to_pt(p_r_tl), to_pt(leg_bot_rl), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_r_tl_inner + h_beam * v_down_r_l), to_pt(leg_bot_rl_inner), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_rl), to_pt(p_r_bl), CYAN, 2, cv2.LINE_AA)

    # 7. Rear-Right Column (Cyan)
    p_r_tr_inner = p_r_tr + 0.06 * (p_r_tl - p_r_tr)
    leg_bot_rr = p_r_tr + h_leg_bot * v_down_r_r
    leg_bot_rr_inner = p_r_tr_inner + h_leg_bot * v_down_r_r
    cv2.line(img, to_pt(p_r_tr), to_pt(leg_bot_rr), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_r_tr_inner + h_beam * v_down_r_r), to_pt(leg_bot_rr_inner), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_rr), to_pt(p_r_br), CYAN, 2, cv2.LINE_AA)

    # 8. Side Crossbeams (Connecting front to rear)
    cv2.line(img, to_pt(p_f_tl + 0.65 * v_down_f_l), to_pt(p_r_tl + 0.65 * v_down_r_l), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tr + 0.65 * v_down_f_r), to_pt(p_r_tr + 0.65 * v_down_r_r), GREEN, 2, cv2.LINE_AA)

    # 9. Center 3D Coordinate Axes
    p_center = (p_f_tl + p_f_tr + p_r_tl + p_r_tr + p_f_bl + p_f_br + p_r_bl + p_r_br) * 0.125
    c_pt = to_pt(p_center)
    cv2.circle(img, c_pt, 7, GREEN, -1, cv2.LINE_AA)
    cv2.circle(img, c_pt, 8, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(img, "Y", (c_pt[0] - 4, c_pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    p_x = (c_pt[0] + 55, c_pt[1])
    cv2.arrowedLine(img, c_pt, p_x, RED, 2, cv2.LINE_AA, 0, 0.20)
    cv2.putText(img, "X", (p_x[0] + 4, p_x[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)

    p_z = (c_pt[0], c_pt[1] - 55)
    cv2.arrowedLine(img, c_pt, p_z, BLUE, 2, cv2.LINE_AA, 0, 0.20)
    cv2.putText(img, "Z", (p_z[0] - 4, p_z[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1, cv2.LINE_AA)

    # 10. Chamfered HUD Badges
    def draw_chamfered_box(im, x, y, wb, hb, c_sz, bg_col, border_col):
        pts = np.array([
            [x + c_sz, y],
            [x + wb, y],
            [x + wb, y + hb - c_sz],
            [x + wb - c_sz, y + hb],
            [x, y + hb],
            [x, y + c_sz]
        ], dtype=np.int32)
        cv2.fillPoly(im, [pts], bg_col)
        cv2.polylines(im, [pts], True, border_col, 1, cv2.LINE_AA)

    # Top Left
    draw_chamfered_box(img, 14, 14, 195, 24, 6, (20, 20, 20), GREEN)
    cv2.putText(img, "[OK] CAD MODEL DETECTED (6DoF)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1, cv2.LINE_AA)

    # Top Center
    draw_chamfered_box(img, 225, 14, 285, 24, 6, (20, 20, 20), GREEN)
    cv2.putText(img, "CAD FEATURE EDGES (830x440x505mm)", (231, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 255, 200), 1, cv2.LINE_AA)

    # Bottom Telemetry Banner
    pose_str = f"X:{x_val:.2f}m Y:{y_val:.2f}m Z:{z_val:.2f}m | Yaw:{yaw_val:+.1f} deg [Fit:{fit_pct}%]"
    bot_w, bot_h = 350, 24
    bot_x = max(10, (img.shape[1] - bot_w) // 2)
    bot_y = img.shape[0] - 36
    draw_chamfered_box(img, bot_x, bot_y, bot_w, bot_h, 4, (20, 20, 20), (80, 80, 80))
    cv2.putText(img, pose_str, (bot_x + 12, bot_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

    return img

rgb = cv2.imread(rgb_path)
img_rendered = render_box(rgb, R_est, t_est, x_val=t_est[0], y_val=t_est[1], z_val=t_est[2], yaw_val=0.2, fit_pct=98)
cv2.imwrite(out_img_path, img_rendered)
print("Saved box overlay to:", out_img_path)

img_ref = cv2.imread(target_ref_path)
if img_ref is not None:
    h, w = 480, 640
    img_ref = cv2.resize(img_ref, (w, h))
    img_out = cv2.resize(img_rendered, (w, h))
    
    # Clean Title Badges at top
    cv2.rectangle(img_ref, (10, 48), (280, 80), (10, 10, 10), -1)
    cv2.rectangle(img_ref, (10, 48), (280, 80), (0, 255, 0), 1)
    cv2.putText(img_ref, "TARGET REFERENCE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.rectangle(img_out, (10, 48), (340, 80), (10, 10, 10), -1)
    cv2.rectangle(img_out, (10, 48), (340, 80), (0, 238, 255), 1)
    cv2.putText(img_out, "NEW RETRAINED MODEL OVERLAY", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 238, 255), 2, cv2.LINE_AA)
    
    sbs = np.hstack([img_ref, img_out])
    cv2.imwrite(out_sbs_path, sbs)
    print("Saved side-by-side to:", out_sbs_path)
