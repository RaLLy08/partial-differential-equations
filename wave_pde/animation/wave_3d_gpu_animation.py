"""
3D Wave Equation Visualization using VisPy (OpenGL) + PyTorch GPU simulation.

The wave amplitude is rendered as the Z-axis of a 3D surface plot.
Click on the surface to add Gaussian pulses.
Camera can be rotated/zoomed with mouse (right-click drag / scroll).

Requirements:
    pip install vispy torch numpy PyQt5
"""

import numpy as np
import torch
import torch.nn as nn
from vispy import app, scene, visuals
from vispy.color import BaseColormap

from animation.initial_states import InitialStates2D
from pulse import get_gaussian_pulse

# ──────────────────────────────────────────────
#  GPU setup
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ──────────────────────────────────────────────
#  Simulation parameters
# ──────────────────────────────────────────────
state = InitialStates2D()

MAX_WAVE_AMPLITUDE = state.MAX_AMPLITUDE
N = state.N
WINDOW_SIZE = state.WINDOW_SIZE
STEPS_PER_FRAME = state.STEPS_PER_FRAME
C = state.C

if not state.is_stable:
    print(f"Warning: CFL condition violated. Simulation may be unstable. C={C:.4f}")

# ──────────────────────────────────────────────
#  Simulation state (GPU tensors)
# ──────────────────────────────────────────────
dim = (N, N)
u_gpu = torch.from_numpy(state.initial_u.astype(np.float32)).to(device)
u_gpu_prev = torch.zeros(dim, device=device)

kernel_gpu = torch.tensor(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]],
    dtype=torch.float32,
    device=device,
).view(1, 1, 3, 3)


def propagate_steps(steps: int):
    """Advance the wave equation by *steps* timesteps on GPU."""
    global u_gpu, u_gpu_prev

    for _ in range(steps):
        laplacian = nn.functional.conv2d(
            u_gpu.unsqueeze(0).unsqueeze(0),
            kernel_gpu,
            padding=1,
        ).squeeze()

        u_gpu_next = 2 * u_gpu - u_gpu_prev + C ** 2 * laplacian
        u_gpu_prev = u_gpu
        u_gpu = u_gpu_next


def add_pulse(gx: int, gy: int):
    """Inject a Gaussian pulse centred at grid coordinates (gx, gy)."""
    global u_gpu

    radius = max(1, int(N * 0.03))
    x_min, x_max = max(0, gx - radius), min(N, gx + radius + 1)
    y_min, y_max = max(0, gy - radius), min(N, gy + radius + 1)

    pulse_shape = (y_max - y_min, x_max - x_min)
    if pulse_shape[0] <= 0 or pulse_shape[1] <= 0:
        return

    pulse = torch.from_numpy(
        get_gaussian_pulse(pulse_shape, MAX_WAVE_AMPLITUDE).astype(np.float32)
    ).to(device)

    u_gpu[y_min:y_max, x_min:x_max] += pulse


# ──────────────────────────────────────────────
#  Custom colormap: blue ─ black ─ red
# ──────────────────────────────────────────────
class WaveColormap(BaseColormap):
    glsl_map = """
    vec4 waveColor(float t) {
        float v = 2.0 * t - 1.0;
        float neg = clamp(-v, 0.0, 1.0);
        float pos = clamp(v, 0.0, 1.0);
        float r = clamp(pos * 1.0 + neg * 0.0,  0.0, 1.0);
        float g = clamp(pos * 0.53 + 0.8*(1.0 - abs(v)*0.9), 0.0, 1.0);
        float b = clamp(neg * 0.8 + 0.85*(1.0 - abs(v)), 0.0, 1.0);
        return vec4(r, g, b, 1.0);
    }
    """
    def map(self, t):
        t = np.atleast_1d(t).astype(float)
        v = 2*t - 1
        neg = np.clip(-v, 0, 1); pos = np.clip(v, 0, 1)
        r = np.clip(pos*1.0 + neg*0.0, 0, 1)
        g = np.clip(pos*0.53 + 0.8*(1 - np.abs(v)*0.9), 0, 1)
        b = np.clip(neg*0.8 + 0.85*(1 - np.abs(v)), 0, 1)
        return np.column_stack([r, g, b, np.ones_like(t)])

# ──────────────────────────────────────────────
#  VisPy canvas & 3-D scene
# ──────────────────────────────────────────────
canvas = scene.SceneCanvas(
    keys="interactive",
    size=(WINDOW_SIZE, WINDOW_SIZE),
    title="Wave Equation – 3-D Surface (VisPy/OpenGL + PyTorch)",
    show=True,
)
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(
    elevation=30,
    azimuth=45,
    distance=3.0,
    fov=45,
    center=(0.5, 0.5, 0),
)

# Coordinate grid [0, 1] × [0, 1]
x = np.linspace(0, 1, N, dtype=np.float32)
y = np.linspace(0, 1, N, dtype=np.float32)

# Initial height data
z_data = u_gpu.detach().cpu().numpy().astype(np.float32)

surface = scene.visuals.SurfacePlot(
    x=x,
    y=y,
    z=z_data / MAX_WAVE_AMPLITUDE * 0.3,   # scale Z for nice visuals
    shading="smooth",
    parent=view.scene,
)
surface.cmap = WaveColormap()



# ──────────────────────────────────────────────
#  Animation timer
# ──────────────────────────────────────────────
_frame_count = 0
_last_time = None


def update(event):
    global _frame_count, _last_time

    propagate_steps(STEPS_PER_FRAME)

    # Pull data back to CPU (fast for moderate N)
    z = u_gpu.detach().cpu().numpy().astype(np.float32)

    # Normalise to [0, 1] for the colormap, scale Z for visual height
    max_amp = 60

    z_norm = np.clip(z / max_amp, -1, 1)
    z_visual = z_norm * 0.3  # height scaling factor

    # Update surface mesh data
    surface.set_data(x=x, y=y, z=z_visual)

    # Colour from amplitude: map [-1,1] → [0,1]
    colors_1d = z_norm.ravel() * 0.5 + 0.5
    cmap = WaveColormap()
    colors = cmap.map(colors_1d).astype(np.float32)
    # SurfacePlot uses MeshVisual internally – update face/vertex colors
    surface.mesh_data.set_vertex_colors(colors)
    surface.mesh_data_changed()

    # FPS counter
    _frame_count += 1
    now = event.elapsed
    if _last_time is None or now - _last_time >= 1.0:
        fps = _frame_count / (now - _last_time) if _last_time is not None else 0
        _frame_count = 0
        _last_time = now

    canvas.update()


timer = app.Timer(interval=1 / 60, connect=update, start=True)

# ──────────────────────────────────────────────
#  Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run()