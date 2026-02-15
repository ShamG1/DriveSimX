import numpy as np
from PIL import Image, ImageDraw
import os

WIDTH, HEIGHT = 1000, 1000
LANE_WIDTH_PX = 42.0

def generate_highway(num_lanes, out_dir):
    """Generates Highway assets (one-way straight road, no yellow lines)."""
    os.makedirs(out_dir, exist_ok=True)
    cx, cy = WIDTH // 2, HEIGHT // 2
    # One-way road centered at cy
    rw = (num_lanes * LANE_WIDTH_PX) / 2.0

    # 1. Drivable area
    drivable = Image.new('L', (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(drivable)
    # Straight horizontal road (one-way)
    d.rectangle([0, cy - rw, WIDTH, cy + rw], fill=255)
    drivable.save(os.path.join(out_dir, 'drivable.png'))

    # 2. Yellow lines (None for highway)
    yellow = Image.new('L', (WIDTH, HEIGHT), 0)
    yellow.save(os.path.join(out_dir, 'yellowline.png'))

    # 3. Lane dashes
    dashes = Image.new('L', (WIDTH, HEIGHT), 0)
    dd = ImageDraw.Draw(dashes)
    
    # Draw dashed lines between lanes
    dist = WIDTH
    dash_len = 20
    gap_len = 20
    
    # For one-way, we draw dashes at multiples of LANE_WIDTH_PX from the top edge
    top_edge = cy - rw
    for i in range(1, num_lanes):
        y_pos = top_edge + i * LANE_WIDTH_PX
        curr = 0.0
        while curr < dist:
            start_x = curr
            end_x = min(curr + dash_len, dist)
            dd.line([(start_x, y_pos), (end_x, y_pos)], fill=255, width=2)
            curr += dash_len + gap_len

    dashes.save(os.path.join(out_dir, 'lane_dashes.png'))

    # 4. Lane ID (encoded by lane index)
    lane_id = Image.new('L', (WIDTH, HEIGHT), 0)
    lid = ImageDraw.Draw(lane_id)
    
    # One-way IDs: 1 to num_lanes
    for j in range(num_lanes):
        # Lane j (0 is top-most)
        y_start = top_edge + j * LANE_WIDTH_PX
        y_end = y_start + LANE_WIDTH_PX
        val = j + 1
        lid.rectangle([0, y_start, WIDTH, y_end], fill=val)

    lane_id.save(os.path.join(out_dir, 'lane_id.png'))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_dir = os.path.join(base_dir, 'scenarios')

    generate_highway(2, os.path.join(scenarios_dir, 'highway_2lane'))
    generate_highway(4, os.path.join(scenarios_dir, 'highway_4lane'))
    print("Highway assets generated in scenarios/highway_2lane and scenarios/highway_4lane")
