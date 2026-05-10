import numpy as np
from PIL import Image
import os
from transformations import render_object

data = np.load('data.npy', allow_pickle=True).item()

v_pos       = data['v_pos']
v_uvs       = data['v_uvs']
t_pos_idx   = data['t_pos_idx']
k_cam_eye   = data['k_cam_eye'].flatten()
up          = data['k_cam_up'].flatten()
target      = data['k_cam_target'].flatten()
focal       = data['k_f']
plane_h     = data['k_sensor_height']
plane_w     = data['k_sensor_width']
fps         = data['k_fps']
duration    = data['k_duration']
radius      = data['k_cam_radius']

res_h = res_w = 512
n_frames = int(fps * duration)

tex = np.array(Image.open('loony-repeat.png').convert('RGB')).astype(np.float64) / 255.0
H, W = tex.shape[:2]
u_idx = np.clip(np.round(v_uvs[:, 0] * (W - 1)).astype(int), 0, W - 1)
v_idx = np.clip(np.round((1.0 - v_uvs[:, 1]) * (H - 1)).astype(int), 0, H - 1)
v_clr = tex[v_idx, u_idx]

# Place camera on orbit circle at given radius, in the same direction as k_cam_eye
cam_dir = k_cam_eye / np.linalg.norm(k_cam_eye)
cam_start = cam_dir * radius

os.makedirs('frames_demo2', exist_ok=True)

for i in range(n_frames):
    # Clockwise rotation around Y: apply R_y(-theta) to cam_start
    # R_y(-theta): x' = x*cos(t) - z*sin(t),  z' = x*sin(t) + z*cos(t)
    theta = 2.0 * np.pi * i / n_frames
    ct = np.cos(theta)
    st = np.sin(theta)
    x0, y0, z0 = cam_start
    cam_eye = np.array([x0 * ct - z0 * st,
                        y0,
                        x0 * st + z0 * ct])

    img = render_object(v_pos, v_clr, t_pos_idx,
                        plane_h, plane_w, res_h, res_w,
                        focal, cam_eye, up, target)

    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(f'frames_demo2/frame_{i:03d}.png')
    print(f'[demo2] frame {i + 1}/{n_frames}')

print('Demo 2 done.')
