import numpy as np
from PIL import Image
import os
from transformations import Trafo, render_object

data = np.load('data.npy', allow_pickle=True).item()

v_pos = data['v_pos']
v_uvs = data['v_uvs']
t_pos_idx = data['t_pos_idx']
eye = data['k_cam_eye'].flatten()
up = data['k_cam_up'].flatten()
target = data['k_cam_target'].flatten()
focal = data['k_f']
plane_h = data['k_sensor_height']
plane_w = data['k_sensor_width']
fps = data['k_fps']
duration = data['k_duration']

res_h = res_w = 512
n_frames = int(fps * duration)

# Sample diffuse texture at each vertex UV to get per-vertex colors for Gouraud shading
tex = np.array(Image.open('loony-repeat.png').convert('RGB')).astype(np.float64) / 255.0
H, W = tex.shape[:2]
u_idx = np.clip(np.round(v_uvs[:, 0] * (W - 1)).astype(int), 0, W - 1)
v_idx = np.clip(np.round((1.0 - v_uvs[:, 1]) * (H - 1)).astype(int), 0, H - 1)
v_clr = tex[v_idx, u_idx]

os.makedirs('frames_demo1', exist_ok=True)

for i in range(n_frames):
    angle = 2.0 * np.pi * i / n_frames

    trafo = Trafo()
    trafo.rotate(np.array([0.0, 1.0, 0.0]), angle, np.array([0.0, 0.0, 0.0]))
    v_pos_rot = trafo.xform_pnts(v_pos)

    img = render_object(v_pos_rot, v_clr, t_pos_idx,
                        plane_h, plane_w, res_h, res_w,
                        focal, eye, up, target)

    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(f'frames_demo1/frame_{i:03d}.png')
    print(f'[demo1] frame {i + 1}/{n_frames}')

print('Demo 1 done.')
