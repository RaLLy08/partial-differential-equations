import numpy as np
import pygame
import time

import torch
import torch.nn as nn

device = torch.device("cuda")

print('Using device:', device)


MAX_TEMP_LABEL = 2000
N = 1000
WINDOW_SIZE = 800

# Initialize
u = np.zeros((N, N), dtype=np.float32)
u[int(N * 0.4):int(N * 0.6), int(N * 0.4):int(N * 0.6)] = MAX_TEMP_LABEL

dt = 0.1
dx = 1
alpha = 2
STEPS_PER_FRAME = 20

gamma = (alpha * dt) / (dx**2)
if gamma > 0.5:
    print(f"Warning: Simulation may be unstable. Gamma: {gamma:.4f}")


kernel_gpu = torch.tensor([
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0]
], device=device, dtype=torch.float32).view(1, 1, 3, 3)
u_gpu = torch.tensor(u, device=device, dtype=torch.float32).view(1, 1, N, N)


def diffuse_steps(steps):
    global u_gpu
    for _ in range(steps):
        # u[1:-1, 1:-1] += gamma * (
        #     u[2:, 1:-1] + u[:-2, 1:-1] +
        #     u[1:-1, 2:] + u[1:-1, :-2] -
        #     4 * u[1:-1, 1:-1]
        # )
      u_processed = nn.functional.conv2d(
        u_gpu,
        kernel_gpu,
        padding=1
      )

      u_gpu += u_processed.squeeze() * gamma


def temperature_to_color(temp_array):
    normalized = np.clip(temp_array / MAX_TEMP_LABEL, 0, 1)

    r = np.clip(normalized * 3, 0, 1)
    g = np.clip(normalized * 3 - 1, 0, 1)
    b = np.clip(normalized * 3 - 2, 0, 1)

    rgb = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    return rgb


def add_heat(x, y):
    global u_gpu

    gx = int(x * N / WINDOW_SIZE)
    gy = int(y * N / WINDOW_SIZE)
    radius = int(N * 0.02)
    x_min = max(1, gx - radius)
    x_max = min(N - 1, gx + radius + 1)
    y_min = max(1, gy - radius)
    y_max = min(N - 1, gy + radius + 1)
    u_gpu[0, 0, y_min:y_max, x_min:x_max] = MAX_TEMP_LABEL


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Heat Equation - GPU (PyTorch)")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 30)
    running = True
    mouse_pressed = False

    while running:
        start_time = time.perf_counter()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = False

        u_cpu = u_gpu.reshape(N, N).detach().cpu().numpy()

        # Add heat while mouse is pressed
        if mouse_pressed:
            mx, my = pygame.mouse.get_pos()
            add_heat(mx, my)


        # Run simulation steps
        diffuse_steps(STEPS_PER_FRAME)

        # Render
        rgb = temperature_to_color(u_cpu)

        # Create surface and scale to window
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (WINDOW_SIZE, WINDOW_SIZE))
        screen.blit(surf, (0, 0))

        # Show FPS
        fps_text = font.render(f"FPS: {clock.get_fps():.0f}, Material Conductivity: {alpha}, Grid Size: {N}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
