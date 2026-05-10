import numpy as np


def vector_interp(p1, p2, V1, V2, coord, dim):
    c1 = p1[dim - 1]
    c2 = p2[dim - 1]
    if c1 == c2:
        return np.array(V1)
    t = (coord - c1) / (c2 - c1)
    return np.array(V1) + t * (np.array(V2) - np.array(V1))


def g_shading(img, vertices, vcolors):
    img = img.copy()
    M, N, _ = img.shape

    v0, v1, v2 = vertices
    c0, c1, c2 = vcolors[0], vcolors[1], vcolors[2]

    ymin = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
    ymax = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), M - 1)

    edges = [(v0, v1, c0, c1), (v1, v2, c1, c2), (v2, v0, c2, c0)]

    for y in range(ymin, ymax + 1):
        intersections = []
        colors_at = []

        for p1, p2, col1, col2 in edges:
            if p1[1] == p2[1]:
                continue
            if (y >= min(p1[1], p2[1])) and (y <= max(p1[1], p2[1])):
                x = vector_interp(p1, p2, p1[0], p2[0], y, 2)
                c_interp = vector_interp(p1, p2, col1, col2, y, 2)
                intersections.append(x)
                colors_at.append(c_interp)

        if len(intersections) < 2:
            continue

        order = np.argsort(intersections)
        x_left  = intersections[order[0]]
        x_right = intersections[order[-1]]
        c_left  = colors_at[order[0]]
        c_right = colors_at[order[-1]]

        x_start = max(int(np.ceil(x_left)), 0)
        x_end   = min(int(np.floor(x_right)), N - 1)

        if x_start > x_end:
            continue

        for x in range(x_start, x_end + 1):
            color = vector_interp((x_left, y), (x_right, y), c_left, c_right, x, 1)
            img[y, x] = np.clip(color, 0, 1)

    return img
