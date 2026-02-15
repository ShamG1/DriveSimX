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


def generate_cross(num_lanes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cx, cy = WIDTH // 2, HEIGHT // 2
    rw = num_lanes * LANE_WIDTH_PX
    cr = CORNER_RADIUS
    stop_off = rw + cr

    drivable = Image.new('L', (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(drivable)

    d.rectangle([cx - rw, 0, cx + rw, HEIGHT], fill=255)
    d.rectangle([0, cy - rw, WIDTH, cy + rw], fill=255)

    d.rectangle([cx - rw - cr, cy - rw - cr, cx - rw, cy - rw], fill=255)
    d.rectangle([cx + rw, cy - rw - cr, cx + rw + cr, cy - rw], fill=255)
    d.rectangle([cx - rw - cr, cy + rw, cx - rw, cy + rw + cr], fill=255)
    d.rectangle([cx + rw, cy + rw, cx + rw + cr, cy + rw + cr], fill=255)

    k = rw + cr + 2.5

    grass_centers = [
        (cx - k, cy - k),
        (cx + k, cy - k),
        (cx - k, cy + k),
        (cx + k, cy + k),
    ]
    for gx, gy in grass_centers:
        d.ellipse([gx - cr, gy - cr, gx + cr, gy + cr], fill=0)

    drivable.save(os.path.join(out_dir, 'drivable.png'))

    yellow = Image.new('L', (WIDTH, HEIGHT), 0)
    y = ImageDraw.Draw(yellow)
    y.rectangle([cx - 1, 0, cx + 1, cy - stop_off], fill=255)
    y.rectangle([cx - 1, cy + stop_off, cx + 1, HEIGHT], fill=255)
    y.rectangle([0, cy - 1, cx - stop_off, cy + 1], fill=255)
    y.rectangle([cx + stop_off, cy - 1, WIDTH, cy + 1], fill=255)
    yellow.save(os.path.join(out_dir, 'yellowline.png'))

    dashes = Image.new('L', (WIDTH, HEIGHT), 0)
    dd = ImageDraw.Draw(dashes)
    for i in range(1, num_lanes):
        off = i * LANE_WIDTH_PX
        draw_dashed_line(dd, (cx - off, 0), (cx - off, cy - stop_off), 255)
        draw_dashed_line(dd, (cx + off, 0), (cx + off, cy - stop_off), 255)
        draw_dashed_line(dd, (cx - off, cy + stop_off), (cx - off, HEIGHT), 255)
        draw_dashed_line(dd, (cx + off, cy + stop_off), (cx + off, HEIGHT), 255)

        draw_dashed_line(dd, (0, cy - off), (cx - stop_off, cy - off), 255)
        draw_dashed_line(dd, (0, cy + off), (cx - stop_off, cy + off), 255)
        draw_dashed_line(dd, (cx + stop_off, cy - off), (WIDTH, cy - off), 255)
        draw_dashed_line(dd, (cx + stop_off, cy + off), (WIDTH, cy + off), 255)

    dashes.save(os.path.join(out_dir, 'lane_dashes.png'))

    lane_id = Image.new('L', (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(lane_id).bitmap((0, 0), drivable, fill=1)
    lane_id.save(os.path.join(out_dir, 'lane_id.png'))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_dir = os.path.join(base_dir, 'scenarios')

    generate_cross(2, os.path.join(scenarios_dir, 'cross_2lane'))
    generate_cross(3, os.path.join(scenarios_dir, 'cross_3lane'))
