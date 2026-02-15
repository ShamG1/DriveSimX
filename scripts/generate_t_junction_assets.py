import numpy as np
from PIL import Image, ImageDraw
import os

WIDTH, HEIGHT = 1000, 1000
LANE_WIDTH_PX = 42.0
CORNER_RADIUS = 84.0


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


def generate_t_junction(num_lanes, out_dir):
    """Generates T-junction assets by removing the North branch."""
    os.makedirs(out_dir, exist_ok=True)
    cx, cy = WIDTH // 2, HEIGHT // 2
    rw = num_lanes * LANE_WIDTH_PX
    cr = CORNER_RADIUS
    stop_off = rw + cr

    # 1. Drivable area
    drivable = Image.new('L', (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(drivable)

    # Main horizontal road (East-West)
    d.rectangle([0, cy - rw, WIDTH, cy + rw], fill=255)
    # Vertical road (South only)
    d.rectangle([cx - rw, cy + rw, cx + rw, HEIGHT], fill=255)
    
    # Fill the intersection core
    d.rectangle([cx - rw, cy - rw, cx + rw, cy + rw], fill=255)

    # Curved corners for T-junction (South-West and South-East)
    d.rectangle([cx - rw - cr, cy + rw, cx - rw, cy + rw + cr], fill=255)
    d.rectangle([cx + rw, cy + rw, cx + rw + cr, cy + rw + cr], fill=255)

    k = rw + cr + 2.5
    # Only two rounded grass corners at the bottom
    grass_centers = [
        (cx - k, cy + k),
        (cx + k, cy + k),
    ]
    for gx, gy in grass_centers:
        d.ellipse([gx - cr, gy - cr, gx + cr, gy + cr], fill=0)

    drivable.save(os.path.join(out_dir, 'drivable.png'))

    # 2. Yellow lines (Center lines)
    yellow = Image.new('L', (WIDTH, HEIGHT), 0)
    y = ImageDraw.Draw(yellow)
    # Horizontal center line (stops at intersection entry)
    y.rectangle([0, cy - 1, cx - stop_off, cy + 1], fill=255)
    y.rectangle([cx + stop_off, cy - 1, WIDTH, cy + 1], fill=255)
    # Vertical center line (South only, stops at intersection entry)
    y.rectangle([cx - 1, cy + stop_off, cx + 1, HEIGHT], fill=255)
    yellow.save(os.path.join(out_dir, 'yellowline.png'))

    # 3. Lane dashes
    dashes = Image.new('L', (WIDTH, HEIGHT), 0)
    dd = ImageDraw.Draw(dashes)
    for i in range(1, num_lanes):
        off = i * LANE_WIDTH_PX
        # East (right branch)
        draw_dashed_line(dd, (cx + stop_off, cy - off), (WIDTH, cy - off), 255)
        draw_dashed_line(dd, (cx + stop_off, cy + off), (WIDTH, cy + off), 255)
        
        # South (bottom branch)
        draw_dashed_line(dd, (cx - off, cy + stop_off), (cx - off, HEIGHT), 255)
        draw_dashed_line(dd, (cx + off, cy + stop_off), (cx + off, HEIGHT), 255)
        
        # West (left branch)
        draw_dashed_line(dd, (0, cy - off), (cx - stop_off, cy - off), 255)
        draw_dashed_line(dd, (0, cy + off), (cx - stop_off, cy + off), 255)

    dashes.save(os.path.join(out_dir, 'lane_dashes.png'))

    # 4. Lane ID (encoded by lane index)
    lane_id = Image.new('L', (WIDTH, HEIGHT), 0)
    lid = ImageDraw.Draw(lane_id)
    
    # Updated direction order to match renumbered C++ backend: East=0, South=1, West=2
    
    # East (0): IDs [1, num_lanes]
    for j in range(num_lanes):
        off = (j + 0.5) * LANE_WIDTH_PX
        val = j + 1
        lid.rectangle([cx + stop_off, cy - off - 0.5 * LANE_WIDTH_PX, WIDTH, cy - off + 0.5 * LANE_WIDTH_PX], fill=val)
        # Entry into intersection
        lid.rectangle([cx, cy - off - 0.5 * LANE_WIDTH_PX, cx + stop_off, cy - off + 0.5 * LANE_WIDTH_PX], fill=val)

    # South (1): IDs [num_lanes + 1, 2 * num_lanes]
    for j in range(num_lanes):
        off = (j + 0.5) * LANE_WIDTH_PX
        val = num_lanes + j + 1
        lid.rectangle([cx + off - 0.5 * LANE_WIDTH_PX, cy + stop_off, cx + off + 0.5 * LANE_WIDTH_PX, HEIGHT], fill=val)
        # Entry into intersection
        lid.rectangle([cx + off - 0.5 * LANE_WIDTH_PX, cy, cx + off + 0.5 * LANE_WIDTH_PX, cy + stop_off], fill=val)

    # West (2): IDs [2 * num_lanes + 1, 3 * num_lanes]
    for j in range(num_lanes):
        off = (j + 0.5) * LANE_WIDTH_PX
        val = 2 * num_lanes + j + 1
        lid.rectangle([0, cy + off - 0.5 * LANE_WIDTH_PX, cx - stop_off, cy + off + 0.5 * LANE_WIDTH_PX], fill=val)
        # Entry into intersection
        lid.rectangle([cx - stop_off, cy + off - 0.5 * LANE_WIDTH_PX, cx, cy + off + 0.5 * LANE_WIDTH_PX], fill=val)

    # Core of the intersection (simple fill for connectivity logic if needed, 
    # but usually path gen handles the gap)
    # We'll leave the core as 0 or fill with 255 if the backend requires it for 'drivable' checks,
    # but the lane_id.png specifically needs lane indices.
    
    lane_id.save(os.path.join(out_dir, 'lane_id.png'))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_dir = os.path.join(base_dir, 'scenarios')

    generate_t_junction(2, os.path.join(scenarios_dir, 'T_2lane'))
    generate_t_junction(3, os.path.join(scenarios_dir, 'T_3lane'))
    print("T-junction assets generated in scenarios/T_2lane and scenarios/T_3lane")
