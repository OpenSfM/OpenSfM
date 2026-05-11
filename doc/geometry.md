# Geometric Models

## Coordinate Systems

### Normalized Image Coordinates

The 2D position of a point in images is stored in *normalized image coordinates*. The origin is in the middle of the image. The x coordinate grows to the right and y grows downward. The larger dimension of the image is 1.

For example, all pixels in a 4:3 image are in the intervals `[-0.5, 0.5]` (x) and `[-0.375, 0.375]` (y).

```
 +-----------------------------+
 |                             |
 |                             |
 |                             |
 |              + ------------>|
 |              | (0,0)  (0.5,0)
 |              |              |
 |              v              |
 +-----------------------------+
              (0, 0.5)
```

Normalized coordinates are resolution-independent and give better numerical stability for multi-view geometry algorithms.

### Pixel Coordinates

Many OpenCV functions use *pixel coordinates* where the origin is at the center of the top-left pixel, x grows right, y grows downward. The bottom-right pixel is at `(width - 1, height - 1)`.

The transformation from normalized to pixel coordinates is:

$$
H = \begin{pmatrix}
     \max(w, h) & 0 & \frac{w-1}{2} \\
     0 & \max(w, h) & \frac{h-1}{2} \\
     0 & 0 & 1
    \end{pmatrix}
$$

and its inverse:

$$
H^{-1} = \begin{pmatrix}
          1 & 0 & -\frac{w-1}{2} \\
          0 & 1 & -\frac{h-1}{2} \\
          0 & 0 & \max(w, h)
         \end{pmatrix}
$$

where $w$ and $h$ are the image width and height.

### Upright Coordinates

When taking pictures, a camera may be in portrait or landscape orientation. Most cameras store this in the EXIF `orientation` tag, and OpenSfM applies the appropriate rotation to masks automatically. Note that normalized and pixel coordinates are *not* corrected for upright — they reflect the original image pixel order.

### World Coordinates

Reconstructed 3D points are stored in *world coordinates*, an arbitrary Euclidean reference frame.

When GPS data is available, a topocentric reference frame is used: origin near the ground, X pointing east, Y pointing north, Z pointing up. The origin's latitude, longitude and altitude are stored in `reference_lla.json`.

When GPS is not available, OpenSfM tries to orient the frame so Z is vertical and the ground is near $z = 0$.

### Camera Coordinates

The *camera coordinate* frame has the origin at the optical center, X pointing right, Y pointing down, Z pointing forward. A point in front of the camera has positive Z.

In the 3D reconstruction viewer, axes are Red (x), Green (y), Blue (z).

![Camera coordinate system](images/id-rotation.png)

The camera pose is the rotation and translation converting world coordinates to camera coordinates.

#### Pose Representation

The `Pose` class stores rotation as an **axis-angle vector** (3D):

- The **direction** is the rotation axis.
- The **length** is the rotation angle in radians.

The camera origin in world coordinates is $-R^T t$. Use `shot.pose.get_origin()` / `set_origin()` to read or write camera position directly.

See `opensfm/types.py` (`Reconstruction`, `Shot`) and `opensfm/src/geometry/`.


## Camera Models

Camera models project 3D points in *camera coordinates* $(x, y, z)$ to *normalized image coordinates* $(u, v)$.

### Perspective (`perspective`)

$$
\begin{array}{l}
x_n = \frac{x}{z} \\
y_n = \frac{y}{z} \\
r^2 = x_n^2 + y_n^2 \\
d = 1 + k_1 r^2 + k_2 r^4 \\
u = f\, d\, x_n \\
v = f\, d\, y_n
\end{array}
$$

### Simple Radial (`simple_radial`)

$$
\begin{array}{l}
x_n = \frac{x}{z},\quad y_n = \frac{y}{z} \\
r^2 = x_n^2 + y_n^2 \\
d = 1 + k_1 r^2 \\
u = f_x\, d\, x_n + c_x \\
v = f_y\, d\, y_n + c_y
\end{array}
$$

### Radial (`radial`)

$$
\begin{array}{l}
x_n = \frac{x}{z},\quad y_n = \frac{y}{z} \\
r^2 = x_n^2 + y_n^2 \\
d = 1 + k_1 r^2 + k_2 r^4 \\
u = f_x\, d\, x_n + c_x \\
v = f_y\, d\, y_n + c_y
\end{array}
$$

### Brown (`brown`)

$$
\begin{array}{l}
x_n = \frac{x}{z},\quad y_n = \frac{y}{z} \\
r^2 = x_n^2 + y_n^2 \\
d_r = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \\
d^t_x = 2p_1\, x_n\, y_n + p_2\,(r^2 + 2x_n^2) \\
d^t_y = 2p_2\, x_n\, y_n + p_1\,(r^2 + 2y_n^2) \\
u = f_x\,(d_r\, x_n + d^t_x) + c_x \\
v = f_y\,(d_r\, y_n + d^t_y) + c_y
\end{array}
$$

### Fisheye (`fisheye`)

$$
\begin{array}{l}
r^2 = x^2 + y^2 \\
\theta = \arctan(r / z) \\
d = 1 + k_1\theta^2 + k_2\theta^4 \\
u = f\, d\, \theta\, \frac{x}{r} \\
v = f\, d\, \theta\, \frac{y}{r}
\end{array}
$$

### Fisheye OpenCV (`fisheye_opencv`)

$$
\begin{array}{l}
r = \sqrt{x^2 + y^2} \\
\theta = \arctan(r / z) \\
d_r = 1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8 \\
u = f\,(d_r\, \theta\, \frac{x}{r}) + c_x \\
v = f\,(d_r\, \theta\, \frac{y}{r}) + c_y
\end{array}
$$

### Fisheye 62 (`fisheye62`)

$$
\begin{array}{l}
r = \sqrt{x^2 + y^2} \\
\theta = \arctan(r / z) \\
d_r = 1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8 + k_5\theta^{10} + k_6\theta^{12} \\
d^t_x = 2p_1\, x_n\, y_n + p_2\,(\theta^2 + 2x_n^2) \\
d^t_y = 2p_2\, x_n\, y_n + p_1\,(\theta^2 + 2y_n^2) \\
u = f\,(d_r\, \theta\, \frac{x}{r} + d^t_x) + c_x \\
v = f\,(d_r\, \theta\, \frac{y}{r} + d^t_y) + c_y
\end{array}
$$

### Fisheye 624 (`fisheye624`)

$$
\begin{array}{l}
r = \sqrt{x^2 + y^2} \\
\theta = \arctan(r / z) \\
d_r = 1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8 + k_5\theta^{10} + k_6\theta^{12} \\
d^t_x = 2p_1\, x_n\, y_n + p_2\,(\theta^2 + 2x_n^2) \\
d^t_y = 2p_2\, x_n\, y_n + p_1\,(\theta^2 + 2y_n^2) \\
d^s_x = s_0\, \theta^2 + s_1\, \theta^4 \\
d^s_y = s_2\, \theta^2 + s_3\, \theta^4 \\
u = f\,(d_r\, \theta\, \frac{x}{r} + d^t_x + d^s_x) + c_x \\
v = f\,(d_r\, \theta\, \frac{y}{r} + d^t_y + d^s_y) + c_y
\end{array}
$$

### Spherical / Equirectangular (`spherical` or `equirectangular`)

$$
\begin{array}{l}
\mathrm{lon} = \arctan\!\left(\frac{x}{z}\right) \\
\mathrm{lat} = \arctan\!\left(\frac{-y}{\sqrt{x^2 + z^2}}\right) \\
u = \frac{\mathrm{lon}}{2\pi} \\
v = -\frac{\mathrm{lat}}{2\pi}
\end{array}
$$

### Dual (`dual`)

$$
\begin{array}{l}
r^2 = x^2 + y^2 \\
x^n_p = \frac{x}{z},\quad y^n_p = \frac{y}{z} \\
x^n_f = \theta\, \frac{x}{r},\quad y^n_f = \theta\, \frac{y}{r} \\
d = 1 + k_1\theta^2 + k_2\theta^4 \\
u = f\, d\,(l\, x^n_p + (1-l)\, x^n_f) \\
v = f\, d\,(l\, y^n_p + (1-l)\, y^n_f)
\end{array}
$$
