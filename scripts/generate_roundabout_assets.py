import numpy as np
from PIL import Image, ImageDraw
import os

WIDTH, HEIGHT = 1000, 1000
LANE_WIDTH_PX = 42.0
INNER_RADIUS = 150.0
FILLET_RADIUS = 255.0


def draw_dashed_line(draw, p1, p2, color, width=2, dash_len=20, gap_len=20):
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist < 1e-6:
        return
    dx = (p2[0] - p1[0]) / dist
    dy = (p2[1] - p1[1]) / dist

    curr = 0.0
    while curr < dist:
        start = (p1[0] + dx * curr, p1[1] + dy * curr)
        end_dist = min(curr + dash_len, dist)
        end = (p1[0] + dx * end_dist, p1[1] + dy * end_dist)
        draw.line([start, end], fill=color, width=width)
        curr += dash_len + gap_len


def draw_dashed_arc(draw, center, radius, start_angle, end_angle, color, width=2, dash_len=20, gap_len=20):
    cx, cy = center
    arc_len_per_deg = (2 * np.pi * radius) / 360.0
    if arc_len_per_deg < 1e-6:
        return

    dash_deg = dash_len / arc_len_per_deg
    gap_deg = gap_len / arc_len_per_deg
    step_deg = dash_deg + gap_deg

    s, e = start_angle, end_angle
    if e < s:
        e += 360.0

    curr_a = s
    while curr_a < e:
        seg_end = min(curr_a + dash_deg, e)
        draw.arc([cx - radius, cy - radius, cx + radius, cy + radius], curr_a, seg_end, fill=color, width=width)
        curr_a += step_deg


def generate_roundabout(num_lanes, out_dir, debug=False):
    os.makedirs(out_dir, exist_ok=True)
    cx, cy = WIDTH // 2, HEIGHT // 2
    rw = num_lanes * LANE_WIDTH_PX
    outer_r = INNER_RADIUS + rw
    fr = FILLET_RADIUS

    # 1. Geometry Calculations
    # Shift corner circles outward and increase radius to enlarge the corner area
    corner_shift = 1.0
    corner_fr = fr + corner_shift

    term = (outer_r + fr) ** 2 - (rw + fr) ** 2
    dy_offset = float(np.sqrt(max(0.0, term)))

    def get_corner_geometry(kx, ky, axis, line_val):
        # T_line: tangent point on arm boundary
        t_line = (line_val, ky) if axis == 'x' else (kx, line_val)
        
        # T_ring: tangent point on roundabout outer circle
        vx, vy = cx - kx, cy - ky
        vnorm = float(np.hypot(vx, vy))
        t_ring = (kx + corner_fr * vx / vnorm, ky + corner_fr * vy / vnorm)
        
        # This sharp_corner calculation in helper is kept for tangent logic,
        # but we will calculate a stricter intersection in the main loop for filling.
        if axis == 'x':
            sharp_corner = (line_val, cy - np.sign(cy-ky)*dy_offset)
        else:
            sharp_corner = (cx - np.sign(cx-kx)*dy_offset, line_val)
            
        a_line = float(np.degrees(np.arctan2(t_line[1] - ky, t_line[0] - kx))) % 360.0
        a_ring = float(np.degrees(np.arctan2(t_ring[1] - ky, t_ring[0] - kx))) % 360.0
        
        da = (a_ring - a_line) % 360.0
        a0_1, a1_1 = a_line, a_line + da
        mid_1 = np.radians(0.5 * (a0_1 + a1_1))
        p1 = (kx + corner_fr * np.cos(mid_1), ky + corner_fr * np.sin(mid_1))
        dist_1 = (p1[0] - cx)**2 + (p1[1] - cy)**2
        
        a0_2, a1_2 = a_ring, a_ring + (360.0 - da)
        mid_2 = np.radians(0.5 * (a0_2 + a1_2))
        p2 = (kx + corner_fr * np.cos(mid_2), ky + corner_fr * np.sin(mid_2))
        dist_2 = (p2[0] - cx)**2 + (p2[1] - cy)**2
        
        if dist_1 < dist_2:
            return a0_1, a1_1, t_line, t_ring, sharp_corner
        else:
            return a0_2, a1_2, t_line, t_ring, sharp_corner

    corner_configs = [
        # N-left/right: Move Y up
        ((cx - rw - corner_fr, cy - dy_offset - corner_shift), 'x', cx - rw),  
        ((cx + rw + corner_fr, cy - dy_offset - corner_shift), 'x', cx + rw),
        # S-left/right: Move Y down
        ((cx - rw - corner_fr, cy + dy_offset + corner_shift), 'x', cx - rw),  
        ((cx + rw + corner_fr, cy + dy_offset + corner_shift), 'x', cx + rw),
        # E-top/bottom: Move Y up/down, move X right
        ((cx + dy_offset + corner_shift, cy - rw - corner_fr), 'y', cy - rw),  
        ((cx + dy_offset + corner_shift, cy + rw + corner_fr), 'y', cy + rw),
        # W-top/bottom: Move Y up/down, move X left
        ((cx - dy_offset - corner_shift, cy - rw - corner_fr), 'y', cy - rw),  
        ((cx - dy_offset - corner_shift, cy + rw + corner_fr), 'y', cy + rw),
    ]

    # ---- Drivable ----
    drivable = Image.new('L', (WIDTH, HEIGHT), 0)
    draw_d = ImageDraw.Draw(drivable)

    # Base Road: Ring + Full arms (white)
    draw_d.ellipse([cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r], fill=255)
    draw_d.rectangle([cx - rw, 0, cx + rw, HEIGHT], fill=255)
    draw_d.rectangle([0, cy - rw, WIDTH, cy + rw], fill=255)

    # ==========================================
    # MODIFIED SECTION: Fill the corner gaps
    # ==========================================
    for (kx, ky), axis, val in corner_configs:
        a0, a1, t_line, t_ring, _ = get_corner_geometry(kx, ky, axis, val)
        
        steps = 30
        arc_pts = []
        for i in range(steps + 1):
            ang = np.radians(a0 + (a1 - a0) * i / steps)
            arc_pts.append((kx + corner_fr * np.cos(ang), ky + corner_fr * np.sin(ang)))
            
        # 1. Calculate the TRUE Sharp Intersection Point (Road Edge vs Outer Ring)
        # This point is "deeper" in the corner than the tangent intersection
        sharp_pt = None
        if axis == 'x': # Vertical Road Edge (x = val)
            # Circle Eq: (x-cx)^2 + (y-cy)^2 = R^2
            # Solve for y: y = cy +/- sqrt(R^2 - (val-cx)^2)
            dx = val - cx
            root = np.sqrt(max(0, outer_r**2 - dx**2))
            # If fillet center (ky) is above center, we want the top intersection (y < cy)
            sy = -1 if ky < cy else 1
            sharp_pt = (val, cy + sy * root)
        else: # Horizontal Road Edge (y = val)
            dy = val - cy
            root = np.sqrt(max(0, outer_r**2 - dy**2))
            # If fillet center (kx) is left of center, we want left intersection (x < cx)
            sx = -1 if kx < cx else 1
            sharp_pt = (cx + sx * root, val)

        # 2. Build the polygon to fill: Sharp Intersection -> Tangent Line -> Arc -> Tangent Ring
        # Determine arc direction to connect correctly
        dist_start_line = np.hypot(arc_pts[0][0]-t_line[0], arc_pts[0][1]-t_line[1])
        
        if dist_start_line < 1.0: 
            # Arc order: Line -> Ring
            # Polygon: SharpPt -> LinePt -> ...Arc... -> RingPt -> Close
            fill_poly = [sharp_pt, t_line] + arc_pts + [t_ring]
        else:
            # Arc order: Ring -> Line
            # Polygon: SharpPt -> RingPt -> ...Arc... -> LinePt -> Close
            fill_poly = [sharp_pt, t_ring] + arc_pts + [t_line]

        # 3. Fill with WHITE (255) to expand drivable area
        draw_d.polygon(fill_poly, fill=255)
    # ==========================================
    # END MODIFIED SECTION
    # ==========================================

    # Inner island (Black hole in center)
    draw_d.ellipse([cx - INNER_RADIUS, cy - INNER_RADIUS, cx + INNER_RADIUS, cy + INNER_RADIUS], fill=0)
    drivable.save(os.path.join(out_dir, 'drivable.png'))

    # ---- Yellow line ----
    yellow = Image.new('L', (WIDTH, HEIGHT), 0)
    draw_y = ImageDraw.Draw(yellow)
    draw_y.line([cx, 0, cx, cy - outer_r], fill=255, width=2)
    draw_y.line([cx, cy + outer_r, cx, HEIGHT], fill=255, width=2)
    draw_y.line([cx + outer_r, cy, WIDTH, cy], fill=255, width=2)
    draw_y.line([0, cy, cx - outer_r, cy], fill=255, width=2)
    yellow.save(os.path.join(out_dir, 'yellowline.png'))

    # ---- Lane dashes ----
    dashes = Image.new('L', (WIDTH, HEIGHT), 0)
    draw_dash = ImageDraw.Draw(dashes)
    for i in range(1, num_lanes):
        off = i * LANE_WIDTH_PX
        draw_dashed_line(draw_dash, (cx - off, 0), (cx - off, cy - outer_r), 255)
        draw_dashed_line(draw_dash, (cx + off, 0), (cx + off, cy - outer_r), 255)
        draw_dashed_line(draw_dash, (cx - off, cy + outer_r), (cx - off, HEIGHT), 255)
        draw_dashed_line(draw_dash, (cx + off, cy + outer_r), (cx + off, HEIGHT), 255)
        draw_dashed_line(draw_dash, (0, cy - off), (cx - outer_r, cy - off), 255)
        draw_dashed_line(draw_dash, (0, cy + off), (cx - outer_r, cy + off), 255)
        draw_dashed_line(draw_dash, (cx + outer_r, cy - off), (WIDTH, cy - off), 255)
        draw_dashed_line(draw_dash, (cx + outer_r, cy + off), (WIDTH, cy + off), 255)

        ring_r = INNER_RADIUS + off
        draw_dashed_arc(draw_dash, (cx, cy), ring_r, 0.0, 360.0, 255)

    # Clip dashes to drivable
    d_np = np.array(dashes)
    r_np = np.array(drivable)
    d_np[r_np == 0] = 0
    Image.fromarray(d_np).save(os.path.join(out_dir, 'lane_dashes.png'))

    # ---- Lane id ----
    lane_id = Image.new('L', (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(lane_id).bitmap((0, 0), drivable, fill=1)
    lane_id.save(os.path.join(out_dir, 'lane_id.png'))

    if debug:
        dbg = Image.fromarray(np.stack([np.array(drivable)] * 3, axis=-1).astype(np.uint8))
        dbg_draw = ImageDraw.Draw(dbg)
        for (kx, ky), axis, val in corner_configs:
            a0, a1, _, _, _ = get_corner_geometry(kx, ky, axis, val)
            dbg_draw.arc([kx - fr, ky - fr, kx + fr, ky + fr], a0, a1, fill=(255, 0, 0), width=3)
        dbg.save(os.path.join(out_dir, 'debug_fillet_overlay.png'))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_dir = os.path.join(base_dir, 'scenarios')
    generate_roundabout(2, os.path.join(scenarios_dir, 'roundabout_2lane'), debug=True)
    generate_roundabout(3, os.path.join(scenarios_dir, 'roundabout_3lane'), debug=True)