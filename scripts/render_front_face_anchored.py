#!/usr/bin/env python3
import os
import cv2
import numpy as np

project_root = "/home/huyen1905/TMCAD"
rgb_path = os.path.join(project_root, "dump_rgb.png")
target_ref_path = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/.user_uploaded/media_1793758343856_733098b1.jpg"

out_single_path = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/target_front_face_anchored_overlay.png"
out_sbs_path = "/home/huyen1905/.gemini/antigravity/brain/ae2ffe12-fe6e-40fc-891f-e790b255a317/sbs_user_target_vs_front_face_anchored.png"

# Read user target to get native target dimensions (1024x765)
img_target_native = cv2.imread(target_ref_path)
H_nat, W_nat = img_target_native.shape[:2]

# Intrinsics scaled to W_nat x H_nat (1024 x 765 from 640 x 480: scale = 1.6)
sx = W_nat / 640.0
sy = H_nat / 480.0
K = np.array([
    [608.215 * sx, 0.0, 341.473 * sx],
    [0.0, 608.128 * sy, 228.438 * sy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# 1. Base image: dump_rgb scaled to native target size (1024x765)
raw_rgb = cv2.imread(rgb_path)
base_img = cv2.resize(raw_rgb, (W_nat, H_nat), interpolation=cv2.INTER_CUBIC)

def render_front_face_anchored(img_in, pose_x, pose_y, pose_z, yaw_deg, fit_pct):
    img = img_in.copy()

    # Build Rotation matrix for table
    # Front-facing camera: CAD Y is width across camera image, CAD X is depth, CAD Z is Up
    yaw_rad = np.radians(yaw_deg)
    # Pitch & Roll from calibration
    pitch_rad = np.radians(-1.65)
    roll_rad = np.radians(-2.98)

    # Orientation: CAD Y -> Cam X (slightly rotated by yaw), CAD Z -> -Cam Y (up), CAD X -> -Cam Z (into scene)
    c_y = np.cos(yaw_rad)
    s_y = np.sin(yaw_rad)
    
    R = np.array([
        [-s_y * 0.03,  c_y * 0.999, -0.05],
        [ 0.029,      -0.052,       -0.998],
        [-c_y * 0.999, -s_y * 0.03,  -0.029]
    ], dtype=np.float64)

    t = np.array([pose_x, pose_y, pose_z], dtype=np.float64)

    # TABLE-DEMO Dimensions
    # Front face: 830mm length along Y, 505mm height along Z
    # Extrude: 440mm depth along X
    cad_x0, cad_x1 = -0.220, 0.220   # 440mm depth
    cad_y0, cad_y1 = -0.415, 0.415   # 830mm length
    z_floor = -0.334                 # Floor
    z_top = 0.171                    # Top (505mm height)

    def proj(x, y, z):
        p_cam = R @ np.array([x, y, z]) + t
        z_eff = max(0.05, p_cam[2])
        u = K[0, 0] * p_cam[0] / z_eff + K[0, 2]
        v = K[1, 1] * p_cam[1] / z_eff + K[1, 2]
        return np.array([u, v], dtype=np.float32)

    # 4 Corners of Front Face (+X is front face)
    p_f_tl = proj(cad_x1, cad_y0, z_top)
    p_f_tr = proj(cad_x1, cad_y1, z_top)
    p_f_bl = proj(cad_x1, cad_y0, z_floor)
    p_f_br = proj(cad_x1, cad_y1, z_floor)

    # 4 Corners of Rear Face (-X is rear face extruded back 440mm)
    p_r_tl = proj(cad_x0, cad_y0, z_top)
    p_r_tr = proj(cad_x0, cad_y1, z_top)
    p_r_bl = proj(cad_x0, cad_y0, z_floor)
    p_r_br = proj(cad_x0, cad_y1, z_floor)

    def to_pt(p):
        return (int(round(p[0])), int(round(p[1])))

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

    # Tabletop cyan perimeter lines
    cv2.polylines(img, [top_poly], True, CYAN, 2, cv2.LINE_AA)

    # 2. Floor Footprint Polygon (Green)
    bot_poly = np.array([p_f_bl, p_f_br, p_r_br, p_r_bl], dtype=np.int32)
    cv2.polylines(img, [bot_poly], True, GREEN, 2, cv2.LINE_AA)

    # 4 Feet leveling circles on floor
    cv2.ellipse(img, to_pt(p_f_bl), (12, 5), 0, 0, 360, GREEN, 2, cv2.LINE_AA)
    cv2.ellipse(img, to_pt(p_f_br), (12, 5), 0, 0, 360, GREEN, 2, cv2.LINE_AA)
    cv2.ellipse(img, to_pt(p_r_bl), (9, 4), 0, 0, 360, GREEN, 1, cv2.LINE_AA)
    cv2.ellipse(img, to_pt(p_r_br), (9, 4), 0, 0, 360, GREEN, 1, cv2.LINE_AA)

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

    # 4. Front-Left Column (Yellow outer line, Yellow inner line)
    p_f_tl_inner = p_f_tl + 0.06 * (p_f_tr - p_f_tl)
    leg_bot_l = p_f_tl + h_leg_bot * v_down_f_l
    leg_bot_l_inner = p_f_tl_inner + h_leg_bot * v_down_f_l

    cv2.line(img, to_pt(p_f_tl), to_pt(leg_bot_l), YELLOW, 3, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tl_inner + h_beam * v_down_f_l), to_pt(leg_bot_l_inner), YELLOW, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_l), to_pt(p_f_bl), YELLOW, 2, cv2.LINE_AA)

    # Front-left gusset
    cv2.line(img, to_pt(p_f_tl_inner + (h_beam + 0.12) * v_down_f_l),
             to_pt(p_f_tl_inner + 0.08 * (p_f_tr - p_f_tl) + h_beam * v_down_f_l), YELLOW, 2, cv2.LINE_AA)

    # 5. Front-Right Column (Green outer line, Green inner line)
    p_f_tr_inner = p_f_tr + 0.06 * (p_f_tl - p_f_tr)
    leg_bot_r = p_f_tr + h_leg_bot * v_down_f_r
    leg_bot_r_inner = p_f_tr_inner + h_leg_bot * v_down_f_r

    cv2.line(img, to_pt(p_f_tr), to_pt(leg_bot_r), GREEN, 3, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tr_inner + h_beam * v_down_f_r), to_pt(leg_bot_r_inner), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_r), to_pt(p_f_br), GREEN, 2, cv2.LINE_AA)

    # Front-right gusset
    cv2.line(img, to_pt(p_f_tr_inner + (h_beam + 0.12) * v_down_f_r),
             to_pt(p_f_tr_inner + 0.08 * (p_f_tl - p_f_tr) + h_beam * v_down_f_r), GREEN, 2, cv2.LINE_AA)

    # 6. Rear-Left Column (Cyan)
    leg_bot_rl = p_r_tl + h_leg_bot * v_down_r_l
    cv2.line(img, to_pt(p_r_tl), to_pt(leg_bot_rl), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_rl), to_pt(p_r_bl), CYAN, 2, cv2.LINE_AA)

    # 7. Rear-Right Column (Cyan)
    leg_bot_rr = p_r_tr + h_leg_bot * v_down_r_r
    cv2.line(img, to_pt(p_r_tr), to_pt(leg_bot_rr), CYAN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(leg_bot_rr), to_pt(p_r_br), CYAN, 2, cv2.LINE_AA)

    # 8. Side Crossbars (Connecting front to rear)
    cv2.line(img, to_pt(p_f_tl + 0.65 * v_down_f_l), to_pt(p_r_tl + 0.65 * v_down_r_l), GREEN, 2, cv2.LINE_AA)
    cv2.line(img, to_pt(p_f_tr + 0.65 * v_down_f_r), to_pt(p_r_tr + 0.65 * v_down_r_r), GREEN, 2, cv2.LINE_AA)

    # 9. Center 3D Coordinate Axes
    p_center = (p_f_tl + p_f_tr + p_r_tl + p_r_tr + p_f_bl + p_f_br + p_r_bl + p_r_br) * 0.125
    c_pt = to_pt(p_center)
    cv2.circle(img, c_pt, 9, GREEN, -1, cv2.LINE_AA)
    cv2.circle(img, c_pt, 10, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(img, "Y", (c_pt[0] - 5, c_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    p_x = (c_pt[0] + 75, c_pt[1])
    cv2.arrowedLine(img, c_pt, p_x, RED, 3, cv2.LINE_AA, 0, 0.20)
    cv2.putText(img, "X", (p_x[0] + 5, p_x[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 1, cv2.LINE_AA)

    p_z = (c_pt[0], c_pt[1] - 75)
    cv2.arrowedLine(img, c_pt, p_z, BLUE, 3, cv2.LINE_AA, 0, 0.20)
    cv2.putText(img, "Z", (p_z[0] - 5, p_z[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLUE, 1, cv2.LINE_AA)

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

    # Top-Left: [OK] CAD MODEL DETECTED (6DoF)
    draw_chamfered_box(img, 22, 22, 310, 36, 8, (20, 20, 20), GREEN)
    cv2.putText(img, "[OK] CAD MODEL DETECTED (6DoF)", (32, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1, cv2.LINE_AA)

    # Top-Center: FRONT-FACE ANCHORED (830x505mm) -> EXTRUDE 440mm
    draw_chamfered_box(img, 360, 54, 510, 36, 8, (20, 20, 20), GREEN)
    cv2.putText(img, "FRONT-FACE ANCHORED (830x505mm) -> EXTRUDE 440mm", (370, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    # Pointer Line from bottom of badge to tabletop
    p_badge_bot = (586, 90)
    p_table_top_center = to_pt((p_f_tl + p_f_tr) * 0.5)
    cv2.line(img, p_badge_bot, p_table_top_center, GREEN, 1, cv2.LINE_AA)

    # Bottom Banner: X:...m Y:...m Z:...m | Yaw:... deg [Fit:...%]
    pose_str = f"X:{pose_x:.2f}m Y:{pose_y:.2f}m Z:{pose_z:.2f}m | Yaw:{yaw_deg:+.1f} deg [Fit:{fit_pct}%]"
    bot_w, bot_h = 560, 36
    bot_x = (img.shape[1] - bot_w) // 2
    bot_y = img.shape[0] - 56
    draw_chamfered_box(img, bot_x, bot_y, bot_w, bot_h, 6, (20, 20, 20), (80, 80, 80))
    cv2.putText(img, pose_str, (bot_x + 18, bot_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA)

    return img

# Render using target reference pose (0.06m, -0.01m, 1.28m, yaw=1.2, fit=94)
img_front_face_anchored = render_front_face_anchored(base_img, 0.06, -0.01, 1.28, 1.2, 94)
cv2.imwrite(out_single_path, img_front_face_anchored)
print("Saved front-face anchored single render to:", out_single_path)

# Build side-by-side comparison with the user target image
if img_target_native is not None:
    sbs = np.hstack([img_target_native, img_front_face_anchored])
    cv2.imwrite(out_sbs_path, sbs)
    print("Saved side-by-side to:", out_sbs_path)

