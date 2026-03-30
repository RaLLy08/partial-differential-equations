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

if not state.is_stable:
    print(f"Warning: CFL condition violated. The simulation may be unstable. C={C:.4f}")

dim = (N, N)
u_gpu = torch.from_numpy(
    state.initial_u.astype(np.float32)
).to(device)

u_gpu_prev = torch.zeros(dim, device=device)

kernel_gpu = torch.tensor([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]],
    dtype=torch.float32,
    device=device
).view(1, 1, 3, 3)


def propagate_steps(steps):
    global u_gpu, u_gpu_prev

    for _ in range(steps):
        laplacian = nn.functional.conv2d(
            u_gpu.unsqueeze(0).unsqueeze(0),
            kernel_gpu,
            padding=1
        ).squeeze()

        u_gpu_next = 2 * u_gpu - u_gpu_prev + C**2 * laplacian

        u_gpu_prev = u_gpu.clone()
        u_gpu = u_gpu_next.clone()


def amplitude_to_color(amp_array):
    # Map [-MAX, +MAX] -> [0, 1], blue=negative, black=zero, red=positive
    normalized = np.clip(amp_array / MAX_WAVE_AMPLITUDE, -1, 1)

    r = np.clip(normalized, 0, 1)
    g = np.zeros_like(normalized)
    b = np.clip(-normalized, 0, 1)

    rgb = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    return rgb


def add_pulse(x, y):
    global u_gpu, u_gpu_prev

    gx = int(x * N / WINDOW_SIZE)
    gy = int(y * N / WINDOW_SIZE)
    radius = int(N * 0.03)

    x_min = max(0, gx - radius)
    x_max = min(N, gx + radius + 1)
    y_min = max(0, gy - radius)
    y_max = min(N, gy + radius + 1)

    pulse_shape = (y_max - y_min, x_max - x_min)
    pulse = torch.from_numpy(
        get_gaussian_pulse(pulse_shape, MAX_WAVE_AMPLITUDE).astype(np.float32)
    ).to(device)

    u_gpu[y_min:y_max, x_min:x_max] += pulse


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

        u_cpu = u_gpu.detach().cpu().numpy()
        rgb = amplitude_to_color(u_cpu)

        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (WINDOW_SIZE, WINDOW_SIZE))
        screen.blit(surf, (0, 0))

        fps_text = font.render(
            f"FPS: {clock.get_fps():.0f} C={C:.2f}  N={N}",
            True, (255, 255, 255)
        )
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
