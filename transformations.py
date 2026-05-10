import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from renderer import g_shading


class Trafo:
    """
    Implements an affine transformation matrix
    """
    def __init__(self):
        self.rot_mat = np.eye(3)
        self.t_vec = np.zeros([3])

    def translate(self, t_vec: np.ndarray) -> None:
        """
        Adds the specified translation vector to the object's
        translation vector.
        """
        self.t_vec = self.t_vec + t_vec

    def rotate(self, axis: np.ndarray, angle: float, center: np.ndarray) -> None:
        """
        Calculates a rotation matrix from the specified axis angle components
        and composes it the the object's rotation matrix
        """
        u = axis / np.linalg.norm(axis)
        ux, uy, uz = u
        c, s = np.cos(angle), np.sin(angle)

        R_new = np.array([
            [(1-c)*ux*ux + c,     (1-c)*ux*uy - s*uz, (1-c)*ux*uz + s*uy],
            [(1-c)*uy*ux + s*uz,  (1-c)*uy*uy + c,    (1-c)*uy*uz - s*ux],
            [(1-c)*uz*ux - s*uy,  (1-c)*uz*uy + s*ux, (1-c)*uz*uz + c   ],
        ])

        # Axis passes through center, so the object's position rotates around it
        self.t_vec   = np.dot(R_new, self.t_vec - center) + center
        self.rot_mat = np.dot(R_new, self.rot_mat)

    def xform_pnts(self, pts: np.ndarray) -> np.ndarray:
        """
        Transforms the incoming points w.r.t the object's
        affine transformation
        """
        return np.dot(self.rot_mat, pts) + self.t_vec[:, np.newaxis]

    def sys2sys(self, sys_src: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Converts the incoming points defined in sys_src coordinate frame
        to the coordinate frame specified by the object's rotation matrix
        and an origin defined by t_vec.
        """
        # src frame -> WCS, then WCS -> destination frame (R^T = R^-1 for orthonormal R)
        pts_world = np.dot(sys_src, pts)
        return np.dot(self.rot_mat.T, pts_world - self.t_vec[:, np.newaxis])


def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray):
    """
    Calculate the camera's view matrix (i.e., its coordinate frame transformation
    specified by a rotation matrix R, and a translation vector t).
    :return a tuple containing the rotation matrix R (3 x 3) and a translation
    vector t (1 x 3)
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # right = up x forward gives the correct +X direction for a right-handed
    # system where camera Z = forward
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    up_cam = np.cross(forward, right)

    R = np.array([right, up_cam, forward])
    t = -np.dot(R, eye)
    return R, t


def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray):
    """
    Project the specified 3d points pts on the image plane, according to a pinhole
    perspective projection model.
    """
    p_cam = np.dot(R, pts) + t[:, np.newaxis]
    depth = p_cam[2]
    pts_2d = focal * p_cam[:2] / p_cam[2:3]
    return pts_2d, depth


def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int) -> np.ndarray:
    """
    Rasterize the incoming 2d points from the camera plane to image pixel
    coordinates
    """
    # Camera plane is centered at origin; image origin is top-left with y flipped
    x_pix = (pts_2d[0] / plane_w + 0.5) * res_w
    y_pix = (0.5 - pts_2d[1] / plane_h) * res_h
    return np.round(np.stack([x_pix, y_pix], axis=1)).astype(int)


def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w,
                  focal, eye, up, target) -> np.ndarray:
    """
    render the specified object from the specified camera.
    """
    R, t = lookat(eye, up, target)
    pts_2d, depth = perspective_project(v_pos, focal, R, t)
    vertices_px = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)

    img = np.ones((res_h, res_w, 3), dtype=np.float64)

    avg_depth = np.mean(depth[t_pos_idx], axis=1)
    order = np.argsort(avg_depth)[::-1]

    for idx in order:
        face = t_pos_idx[idx]
        img = g_shading(img, vertices_px[face], v_clr[face])

    return img
