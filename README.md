# Partial Differential Equations

Interactive simulations and notebooks for exploring PDEs numerically. Includes GPU-accelerated real-time animations and Jupyter notebooks for experimentation.


## File Descriptions

### `wave_pde/pulse.py`
Utility functions for generating initial wave disturbances:
- `get_gaussian_pulse` — creates a smooth bell-curve pulse in 1D or 2D, parameterized by amplitude, sigma (spread), and center position.
- `get_square_pulse` — creates a flat-top pulse over a centered region.

Run directly to preview both 1D and 2D pulse shapes via matplotlib.

---

### `wave_pde/animation/initial_states.py`
Frozen dataclasses that define simulation parameters and initial wave states:
- `WaveStateBase` — shared base: grid size `N`, spacing `dx`, timestep `dt`, wave speed `c`, and Courant number `C`.
- `InitialStates1D` — 1D config (N=100, c=8). Validates the CFL stability condition `C ≤ 1`.
- `InitialStates2D` — 2D config (N=600, c=12, window 800px). Validates `C ≤ 1/√2`. Initial field starts flat; pulses are added interactively at runtime.

---

### `wave_pde/animation/wave_1d_animation.py`
Simulates the 1D wave equation using finite differences on CPU (numpy). Renders with matplotlib using a colored `LineCollection` (plasma colormap mapped to amplitude). Runs 200 frames with 25 propagation steps per frame.

---

### `wave_pde/animation/wave_2d_gpu_animation.py`
Real-time 2D wave simulation on GPU using PyTorch + CUDA. The discrete Laplacian is computed via 2D convolution. Renders to a pygame window with a red/black/blue colormap (positive/zero/negative amplitude). Interactive: click and drag to inject Gaussian wave pulses anywhere on the grid.


### `wave_pde/animation/wave_3d_gpu_animation.py`
Extends the 2D GPU simulation into a 3D surface plot rendered with VisPy (OpenGL). Wave amplitude maps to the Z-axis height of a mesh. Uses a custom GLSL colormap (blue–black–red). Camera is fully interactive.


### `heat_pde/heat-2d_gpu_animation.py`
Real-time 2D heat equation simulation on GPU (PyTorch + CUDA). Diffusion is computed via convolution with the discrete Laplacian kernel. Renders to pygame with a black→red→yellow→white temperature colormap. Click to inject heat sources.



### `wave_pde/wave.ipynb` / `heat_pde/heat.ipynb`
Jupyter notebooks for step-by-step exploration of each PDE: derivation, numerical schemes, stability analysis, and static plots.

---

## Running the Animations

All animation scripts must be run from the `wave_pde/` or `heat_pde/` directory so that relative imports resolve correctly.

### 1D Wave (matplotlib, CPU)
```bash
cd wave_pde
python -m animation.wave_1d_animation
```

### 2D Wave (pygame, CUDA required)
```bash
cd wave_pde
python -m animation.wave_2d_gpu_animation
```

### 3D Wave Surface (VisPy/OpenGL, CUDA preferred)
```bash
cd wave_pde
python -m animation.wave_3d_gpu_animation
```

### 2D Heat Diffusion (pygame, CUDA required)
```bash
cd heat_pde
python heat-2d_gpu_animation.py
```

---

## Requirements

```bash
pip install numpy matplotlib pygame torch vispy PyQt5
```

A CUDA-capable GPU is required for the 2D/3D wave and heat animations. The 3D wave script falls back to CPU if no GPU is detected; the others will error without CUDA.
