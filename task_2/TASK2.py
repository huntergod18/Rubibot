#submission by MASTERBOT team
#RAJ ARYAN 250869
#ABHINAV BAJPAI 250029
import cv2
import numpy as np
import math
import kociemba

SCAN_ORDER     = ["U", "R", "F", "D", "L", "B"]
KOCIEMBA_ORDER = ["U", "R", "F", "D", "L", "B"]

ORIENT = {
    "U": {
        "title":     "WHITE FACE  (U)",
        "step1":     "Show WHITE center to camera.",
        "step2":     "Make sure BLUE is on TOP.",
        "step3":     "Make sure RED is on your LEFT.",
        "top_lbl":   "BLUE (B) at top",
        "left_lbl":  "RED (R) at left",
    },
    "R": {
        "title":     "RED FACE  (R)",
        "step1":     "Show RED center to camera.",
        "step2":     "Make sure WHITE is on TOP.",
        "step3":     "Make sure BLUE is on your LEFT.",
        "top_lbl":   "WHITE (U) at top",
        "left_lbl":  "BLUE (B) at left",
    },
    "F": {
        "title":     "GREEN FACE  (F)",
        "step1":     "Show GREEN center to camera.",
        "step2":     "Make sure WHITE is on TOP.",
        "step3":     "Make sure RED is on your LEFT.",
        "top_lbl":   "WHITE (U) at top",
        "left_lbl":  "RED (R) at left",
    },
    "D": {
        "title":     "YELLOW FACE  (D)",
        "step1":     "Show YELLOW center to camera.",
        "step2":     "Make sure GREEN is on TOP.",
        "step3":     "Make sure RED is on your LEFT.",
        "top_lbl":   "GREEN (F) at top",
        "left_lbl":  "RED (R) at left",
    },
    "L": {
        "title":     "ORANGE FACE  (L)",
        "step1":     "Show ORANGE center to camera.",
        "step2":     "Make sure WHITE is on TOP.",
        "step3":     "Make sure GREEN is on your LEFT.",
        "top_lbl":   "WHITE (U) at top",
        "left_lbl":  "GREEN (F) at left",
    },
    "B": {
        "title":     "BLUE FACE  (B)",
        "step1":     "Show BLUE center to camera.",
        "step2":     "Make sure WHITE is on TOP.",
        "step3":     "Make sure ORANGE is on your LEFT.",
        "top_lbl":   "WHITE (U) at top",
        "left_lbl":  "ORANGE (L) at left",
    },
}

FACE_BGR = {
    "U": (255, 255, 255),   
    "R": (0,   0,   220),   
    "F": (0,   200,   0),   
    "D": (0,   220, 255),   
    "L": (0,   120, 255),  
    "B": (210,  60,   0),   
}

FACE_NAME = {
    "U": "WHITE",
    "R": "RED",
    "F": "GREEN",
    "D": "YELLOW",
    "L": "ORANGE",
    "B": "BLUE",
}

CAM_W, CAM_H  = 640, 480
PANEL_W       = 340    
WIN_W         = PANEL_W + CAM_W
WIN_H         = CAM_H

GRID_STRIDE   = 72
PATCH_HALF    = 12
PREV_CELL     = 38
PREV_MARGIN   = 10

C_BG      = (15,  18,  22)    
C_PANEL2  = (22,  26,  32)    
C_ACCENT  = (255, 200, 0)     
C_TITLE   = (250, 250, 250)   
C_SUB     = (180, 190, 200)   
C_DIM     = (100, 110, 120)   
C_WARN    = (0,   200, 255)   
C_RED     = (80,  80,  240)   

FONT      = cv2.FONT_HERSHEY_SIMPLEX

def get_grid_origin(cam_w, cam_h):
    span = 2 * GRID_STRIDE
    return cam_w // 2 - span // 2, cam_h // 2 - span // 2

def facelet_center(row, col, ox, oy):
    return ox + col * GRID_STRIDE, oy + row * GRID_STRIDE

def extract_patch_hsv(hsv, cx, cy):
    patch = hsv[cy - PATCH_HALF: cy + PATCH_HALF,
                cx - PATCH_HALF: cx + PATCH_HALF]
    if patch.size == 0:
        return [0, 0, 0]
    return np.median(patch.reshape(-1, 3), axis=0).tolist()

def classify_hsv(sample, references):
    h, s, v = sample

    if s < 60 and v > 150:
        return "U"
    if 18 <= h <= 38 and s > 80 and v > 130:
        return "D"

    best_key  = None
    best_dist = float("inf")
    
    for key, ref in references.items():
        if key == "U": 
            continue
            
        rh, rs, rv = ref
        dh = min(abs(h - rh), 180 - abs(h - rh))
        
        dist = math.sqrt(
            20.0 * (dh) ** 2 +
            0.5 * (s - rs) ** 2 +
            0.1 * (v - rv) ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_key  = key
            
    return best_key

def _txt(canvas, txt, x, y, scale=0.50, color=C_TITLE, thick=1):
    cv2.putText(canvas, str(txt), (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)

def draw_mini_cube_diagram(canvas, face_key, ox, oy):
    info  = ORIENT[face_key]
    cell  = 18
    total = 3 * cell

    for r in range(3):
        for c in range(3):
            x1 = ox + c * cell
            y1 = oy + r * cell
            fill_col = FACE_BGR[face_key] if r == 1 and c == 1 else C_DIM
            cv2.rectangle(canvas, (x1, y1), (x1+cell, y1+cell), fill_col, -1)
            cv2.rectangle(canvas, (x1, y1), (x1+cell, y1+cell), (0,0,0), 1)

    cx = ox + total // 2
    cy = oy + total // 2

    top_face = info["top_lbl"].split("(")[1].rstrip(")")
    top_bgr  = FACE_BGR.get(top_face, C_DIM)
    cv2.arrowedLine(canvas, (cx, oy - 4), (cx, oy - 18), top_bgr, 2, tipLength=0.4)
    _txt(canvas, "TOP", ox + total + 4, oy - 10, 0.35, top_bgr)

    left_face = info["left_lbl"].split("(")[1].rstrip(")")
    left_bgr  = FACE_BGR.get(left_face, C_DIM)
    lx_start   = ox - 4
    lx_end     = ox - 18
    cv2.arrowedLine(canvas, (lx_start, cy), (lx_end, cy), left_bgr, 2, tipLength=0.4)
    _txt(canvas, "LEFT", lx_end - 32, cy + 4, 0.35, left_bgr)

def draw_panel(canvas, stage, face_key=None, face_num=0, calibrated=False, warn_blank=False):
    cv2.rectangle(canvas, (0, 0), (PANEL_W, WIN_H), C_BG, -1)
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, WIN_H), C_ACCENT, 2)

    x0 = 20
    x_line = PANEL_W - 20
    y = 40

    _txt(canvas, "RUBIBOT", x0, y, 0.80, C_ACCENT, 2); y += 20
    cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 35

    if stage == "idle":
        _txt(canvas, "WELCOME", x0, y, 0.60, C_TITLE, 1); y += 30
        _txt(canvas, "Place your SOLVED cube", x0, y, 0.50, C_SUB); y += 25
        _txt(canvas, "inside the visual grid.", x0, y, 0.50, C_SUB); y += 25
        _txt(canvas, "Align each circle with", x0, y, 0.50, C_SUB); y += 25
        _txt(canvas, "a sticker center.", x0, y, 0.50, C_SUB); y += 35

        cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 30
        _txt(canvas, "CALIBRATION RULES:", x0, y, 0.50, C_DIM); y += 30
        _txt(canvas, "Point camera at the", x0, y, 0.50, C_WARN); y += 25
        _txt(canvas, "center of each face", x0, y, 0.50, C_WARN); y += 35

        cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 30
        _txt(canvas, "SPACE = start calibration", x0, y, 0.55, C_ACCENT); y += 30
        _txt(canvas, "ESC   = quit", x0, y, 0.55, C_RED)

    elif stage in ("calib", "scan"):
        label = "CALIBRATION" if stage == "calib" else "SCANNING"
        _txt(canvas, label, x0, y, 0.65, C_ACCENT, 2); y += 25

        bar_w = PANEL_W - 40
        filled = int(bar_w * face_num / 6)
        cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + 10), C_DIM, -1)
        cv2.rectangle(canvas, (x0, y), (x0 + filled, y + 10), C_ACCENT, -1)
        y += 35
        _txt(canvas, f"Face {face_num+1} of 6", x0, y, 0.50, C_SUB); y += 20
        cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 30

        if face_key:
            info     = ORIENT[face_key]
            face_bgr = FACE_BGR[face_key]

            cv2.rectangle(canvas, (x0, y - 16), (x0 + 16, y), face_bgr, -1)
            cv2.rectangle(canvas, (x0, y - 16), (x0 + 16, y), (0,0,0), 1)
            _txt(canvas, info["title"], x0 + 26, y, 0.55, C_TITLE, 1); y += 25
            cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 30

            _txt(canvas, "HOW TO HOLD:", x0, y, 0.45, C_DIM); y += 25
            _txt(canvas, info["step1"], x0, y, 0.45, C_SUB);  y += 20
            _txt(canvas, info["step2"], x0, y, 0.45, C_WARN); y += 20
            _txt(canvas, info["step3"], x0, y, 0.45, C_WARN); y += 30

            cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 25
            _txt(canvas, "CAMERA VIEW AXES:", x0, y, 0.45, C_DIM); y += 25
            _txt(canvas, info["top_lbl"], x0 + 5, y, 0.45, C_SUB); y += 20
            _txt(canvas, info["left_lbl"], x0 + 5, y, 0.45, C_SUB); y += 30

            cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 25
            _txt(canvas, "FACE DIAGRAM:", x0, y, 0.45, C_DIM); y += 15
            
            draw_mini_cube_diagram(canvas, face_key, x0 + 40, y); y += 75

            if warn_blank:
                cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 25
                _txt(canvas, "!! BLANK SURFACE !!", x0, y, 0.50, C_RED, 2); y += 25
                _txt(canvas, "Point camera at cube", x0, y, 0.45, C_WARN); y += 20
                _txt(canvas, "center then press SPACE.", x0, y, 0.45, C_WARN)

    elif stage == "between":
        _txt(canvas, "CALIBRATION", x0, y, 0.65, C_ACCENT); y += 30
        _txt(canvas, "COMPLETE!",   x0, y, 0.65, C_ACCENT); y += 35
        cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 35
        
        _txt(canvas, "Now swap to your",    x0, y, 0.50, C_SUB); y += 25
        _txt(canvas, "SCRAMBLED cube.",     x0, y, 0.50, C_SUB); y += 25
        _txt(canvas, "Follow the visual",   x0, y, 0.45, C_DIM); y += 25
        _txt(canvas, "anchors carefully.",  x0, y, 0.45, C_DIM); y += 35
        
        cv2.line(canvas, (x0, y), (x_line, y), C_DIM, 1); y += 35
        _txt(canvas, "SPACE = start scan", x0, y, 0.55, C_ACCENT); y += 30
        _txt(canvas, "ESC   = quit",       x0, y, 0.55, C_RED)

   
    badge_col  = C_ACCENT if calibrated else C_WARN
    badge_text = "CALIBRATED" if calibrated else "NOT CALIBRATED"
    status_dot = "(+)" if calibrated else "(!)"
    cv2.rectangle(canvas, (0, WIN_H - 40), (PANEL_W, WIN_H), C_PANEL2, -1)
    cv2.line(canvas, (0, WIN_H - 40), (PANEL_W, WIN_H - 40), C_DIM, 1)
    _txt(canvas, f"{status_dot} {badge_text}", 20, WIN_H - 15, 0.50, badge_col, 1)

def draw_camera_hud(cam_img, references, face_key=None):
    h, w   = cam_img.shape[:2]
    hsv    = cv2.cvtColor(cam_img, cv2.COLOR_BGR2HSV)
    ox, oy = get_grid_origin(w, h)
    preview = []

    for row in range(3):
        for col in range(3):
            cx, cy = facelet_center(row, col, ox, oy)
            cv2.circle(cam_img, (cx, cy), PATCH_HALF + 4, C_ACCENT, 2)
            if references:
                s   = extract_patch_hsv(hsv, cx, cy)
                lbl = classify_hsv(s, references)
                preview.append(FACE_BGR.get(lbl, (80, 80, 80)))
            else:
                preview.append((60, 60, 60))

    pad = PATCH_HALF + 10
    cv2.rectangle(cam_img,
                  (ox - pad,                 oy - pad),
                  (ox + 2*GRID_STRIDE + pad, oy + 2*GRID_STRIDE + pad),
                  C_ACCENT, 2)

    if face_key:
        info  = ORIENT[face_key]
        total = 2 * GRID_STRIDE
        cx_g  = ox + total // 2
        cy_g  = oy + total // 2

        top_face = info["top_lbl"].split("(")[1].rstrip(")")
        top_bgr  = FACE_BGR.get(top_face, C_DIM)
        ay_start = oy - pad - 6
        ay_end   = oy - pad - 28
        cv2.arrowedLine(cam_img, (cx_g, ay_start), (cx_g, ay_end), top_bgr, 2, tipLength=0.35)
        _txt(cam_img, info["top_lbl"].split()[0], cx_g - 28, ay_end - 4, 0.45, top_bgr, 1)

        left_face = info["left_lbl"].split("(")[1].rstrip(")")
        left_bgr  = FACE_BGR.get(left_face, C_DIM)
        ax_start   = ox - pad - 6
        ax_end     = ox - pad - 28
        cv2.arrowedLine(cam_img, (ax_start, cy_g), (ax_end, cy_g), left_bgr, 2, tipLength=0.35)
        _txt(cam_img, info["left_lbl"].split()[0], ax_end - 45, cy_g + 4, 0.45, left_bgr, 1)

    px0 = w - PREV_MARGIN - 3 * PREV_CELL
    py0 = PREV_MARGIN
    cv2.rectangle(cam_img,
                  (px0-3, py0-3),
                  (px0 + 3*PREV_CELL + 3, py0 + 3*PREV_CELL + 3),
                  C_ACCENT, 1)
    
    for i, bgr in enumerate(preview):
        r  = i // 3; c = i % 3
        x1 = px0 + c * PREV_CELL
        y1 = py0 + r * PREV_CELL
        cv2.rectangle(cam_img, (x1, y1), (x1+PREV_CELL-1, y1+PREV_CELL-1), bgr, -1)
        cv2.rectangle(cam_img, (x1, y1), (x1+PREV_CELL-1, y1+PREV_CELL-1), (0,0,0), 1)
    _txt(cam_img, "LIVE PREVIEW", px0, py0 + 3*PREV_CELL + 16, 0.45, C_TITLE, 1)

def build_frame(cam_frame, references, stage, face_key=None, face_num=0, warn_blank=False):
    cam_r = cv2.resize(cam_frame, (CAM_W, CAM_H))
    draw_camera_hud(cam_r, references, face_key=face_key)
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:, PANEL_W:] = cam_r
    draw_panel(canvas, stage, face_key=face_key, face_num=face_num,
               calibrated=bool(references), warn_blank=warn_blank)
    return canvas

def _is_blank_sample(ref):
    h, s, v = ref
    return s < 30 and v < 150

def run_calibration(camera):
    references = {}
    print("\n===== CALIBRATION =====")
    print("Point the camera at the CENTER sticker, then press SPACE.")

    for idx, face in enumerate(SCAN_ORDER):
        warn_blank = False
        while True:
            ok, frame = camera.read()
            if not ok: continue
            
            frame = cv2.flip(frame, 1) 
            
            canvas = build_frame(frame, references, "calib",
                                 face_key=face, face_num=idx,
                                 warn_blank=warn_blank)
            cv2.imshow("Rubibot", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cv2.destroyAllWindows(); return {}

            if key == 32:
                fh, fw = frame.shape[:2]
                sx, sy = fw / CAM_W, fh / CAM_H
                ox = int(get_grid_origin(CAM_W, CAM_H)[0] * sx)
                oy = int(get_grid_origin(CAM_W, CAM_H)[1] * sy)
                st = int(GRID_STRIDE * sx)
                ph = int(PATCH_HALF  * sx)

                hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cx, cy = ox + st, oy + st
                patch = hsv[cy-ph:cy+ph, cx-ph:cx+ph]
                ref   = np.median(patch.reshape(-1, 3), axis=0).tolist()

                if _is_blank_sample(ref):
                    warn_blank = True
                    continue

                references[face] = ref
                warn_blank = False
                print(f"  {face} ({FACE_NAME[face]}): H={ref[0]:.1f} S={ref[1]:.1f} V={ref[2]:.1f} OK")
                break

    print("\nCalibration complete.")
    return references

def scan_scrambled_cube(camera, references):
    cube = {f: [["?" for _ in range(3)] for _ in range(3)] for f in SCAN_ORDER}

    print("\n===== SCRAMBLED SCAN =====")
    
    for idx, face in enumerate(SCAN_ORDER):
        while True:
            ok, frame = camera.read()
            if not ok: continue
            
            frame = cv2.flip(frame, 1)
            
            canvas = build_frame(frame, references, "scan", face_key=face, face_num=idx)
            cv2.imshow("Rubibot", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cv2.destroyAllWindows(); return ""

            if key == 32:
                fh, fw = frame.shape[:2]
                sx, sy = fw / CAM_W, fh / CAM_H
                ox = int(get_grid_origin(CAM_W, CAM_H)[0] * sx)
                oy = int(get_grid_origin(CAM_W, CAM_H)[1] * sy)
                st = int(GRID_STRIDE * sx)
                ph = int(PATCH_HALF  * sx)

                hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                raw_face = []
                for row in range(3):
                    raw_row = []
                    for col in range(3):
                        cx, cy = ox + col * st, oy + row * st
                        p  = hsv[cy-ph:cy+ph, cx-ph:cx+ph]
                        s  = (np.median(p.reshape(-1, 3), axis=0).tolist() if p.size > 0 else [0, 0, 0])
                        raw_row.append(classify_hsv(s, references))
                        
                    raw_row.reverse() 
                    raw_face.append(raw_row)

                cube[face] = raw_face
                print(f"  {face} ({FACE_NAME[face]}): {cube[face]}")
                break

    cube_string = "".join(
        cube[f][r][c] for f in KOCIEMBA_ORDER for r in range(3) for c in range(3)
    )

    print("\n===== COLOUR COUNT =====")
    valid = True
    for face in KOCIEMBA_ORDER:
        cnt  = cube_string.count(face)
        flag = "OK" if cnt == 9 else "!! WRONG !!"
        print(f"  {face} ({FACE_NAME[face]}): {cnt}  {flag}")
        if cnt != 9: valid = False

    if not valid: print("\n[WARNING] Colour count wrong -- rescan at least one face.")
    return cube_string

def print_solution(solution):
    moves = solution.strip().split()

    sep = "=" * 55
    print("\n" + sep)
    print("  STARTING HOLD ORIENTATION")
    print(sep)
    print("  1. Hold GREEN face directly toward you (Front).")
    print("  2. Hold WHITE face pointing to the ceiling (Top).")
    print(sep)
    print(f"\n  SOLUTION SEQUENCE ({len(moves)} moves):")
    print(f"  {' '.join(moves)}\n")
    print(sep)

SOLVED_STRING = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

def is_already_solved(s):
    return s == SOLVED_STRING

def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Camera not found."); return

    references = {}

    while True:
        ok, frame = camera.read()
        if not ok: continue
        frame = cv2.flip(frame, 1) 
        cv2.imshow("Rubibot", build_frame(frame, references, "idle"))
        key = cv2.waitKey(1) & 0xFF
        if key == 27: camera.release(); cv2.destroyAllWindows(); return
        if key == 32: break
    cv2.destroyAllWindows()

    references = run_calibration(camera)
    if not references: camera.release(); return
    cv2.destroyAllWindows()

    while True:
        ok, frame = camera.read()
        if not ok: continue
        frame = cv2.flip(frame, 1) 
        cv2.imshow("Rubibot", build_frame(frame, references, "between"))
        key = cv2.waitKey(1) & 0xFF
        if key == 27: camera.release(); cv2.destroyAllWindows(); return
        if key == 32: break
    cv2.destroyAllWindows()

    cube_string = scan_scrambled_cube(camera, references)
    camera.release()
    cv2.destroyAllWindows()

    if len(cube_string) != 54:
        print("[ERROR] Invalid cube string length -- please rescan."); return

    if is_already_solved(cube_string):
        print("\n   CUBE IS ALREADY SOLVED!\n"); return

    print("\nSolving with Kociemba...")
    try:
        solution = kociemba.solve(cube_string)
        print_solution(solution)
    except Exception as e:
        print(f"\n[ERROR] Kociemba solver failed: {e}")
        print("\nCommon causes:")
        print("  1. Lighting made colours hard to distinguish.")
        print("  2. A sticker was misread -> check colour count above.")

if __name__ == "__main__":
    main()