import numpy as np
import pygame

import torch
import torch.nn as nn

from animation.initial_states import InitialStates2D
from pulse import get_gaussian_pulse

device = torch.device("cuda")

print('Using device:', device)

state = InitialStates2D()

MAX_WAVE_AMPLITUDE = state.MAX_AMPLITUDE
N = state.N
WINDOW_SIZE = state.WINDOW_SIZE
STEPS_PER_FRAME = state.STEPS_PER_FRAME
C = state.C
C_sq = C ** 2  # pre-compute once

if not state.is_stable:
    print(f"Warning: CFL condition violated. The simulation may be unstable. C={C:.4f}")

dim = (N, N)

# Three pre-allocated GPU buffers — no dynamic allocation in the hot loop
u_prev = torch.zeros(dim, device=device)
u_curr = torch.from_numpy(state.initial_u.astype(np.float32)).to(device)
u_next = torch.zeros(dim, device=device)

kernel_gpu = torch.tensor([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]],
    dtype=torch.float32,
    device=device
).view(1, 1, 3, 3)


@torch.no_grad()
def propagate_steps(steps):
    global u_prev, u_curr, u_next

    for _ in range(steps):
        laplacian = nn.functional.conv2d(
            u_curr.unsqueeze(0).unsqueeze(0),
            kernel_gpu,
            padding=1
        ).squeeze_()

        # Write result into the pre-allocated u_next buffer (no allocation)
        torch.add(u_curr, u_curr, out=u_next)   # u_next = 2 * u_curr
        u_next.sub_(u_prev)                      # u_next -= u_prev
        u_next.add_(laplacian, alpha=C_sq)       # u_next += C² * laplacian

        # Rotate buffers — zero allocations, zero copies
        u_prev, u_curr, u_next = u_curr, u_next, u_prev


@torch.no_grad()
def amplitude_to_color(u_tensor):
    # Compute color mapping on GPU; transfer uint8 (3x smaller than float32)
    normalized = torch.clamp(u_tensor * (1.0 / MAX_WAVE_AMPLITUDE), -1.0, 1.0)

    pos = torch.clamp(normalized, 0.0, 1.0)   # crests  0→1
    neg = torch.clamp(-normalized, 0.0, 1.0)  # troughs 0→1

    # Gamma curve — lifts small-amplitude waves so they stay visible
    pos = torch.pow(pos, 0.6)
    neg = torch.pow(neg, 0.6)

    r = torch.clamp(pos, 0.0, 1.0)
    b = torch.clamp(neg, 0.0, 1.0)
    g = torch.zeros_like(pos)

    rgb = torch.stack([r, g, b], dim=-1).mul_(255).to(torch.uint8)
    return rgb.cpu().numpy()


def add_pulse(x, y):
    global u_curr

    gx = int(x * N / WINDOW_SIZE)
    gy = int(y * N / WINDOW_SIZE)
    radius = int(N * 0.03)

    x_min = max(0, gx - radius)
    x_max = min(N, gx + radius + 1)
    y_min = max(0, gy - radius)
    y_max = min(N, gy + radius + 1)

    pulse_shape = (y_max - y_min, x_max - x_min)
    pulse = torch.from_numpy(
        get_gaussian_pulse(pulse_shape, 0.1, 4).astype(np.float32)
    ).to(device)

    u_curr[y_min:y_max, x_min:x_max].add_(pulse)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Wave Equation - GPU (PyTorch)")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 30)
    running = True
    mouse_pressed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = False

        if mouse_pressed:
            mx, my = pygame.mouse.get_pos()
            add_pulse(mx, my)

        propagate_steps(STEPS_PER_FRAME)

        rgb = amplitude_to_color(u_curr)

        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if N != WINDOW_SIZE:
            surf = pygame.transform.scale(surf, (WINDOW_SIZE, WINDOW_SIZE))
        screen.blit(surf, (0, 0))

        # fps_text = font.render(
        #     f"FPS: {clock.get_fps():.0f} C={C:.2f}  N={N}",
        #     True, (255, 255, 255)
        # )
        # screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
