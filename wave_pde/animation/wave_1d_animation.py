import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from animation.initial_states import InitialStates1D

initial_states = InitialStates1D()


u = initial_states.initial_u
N = initial_states.N
MAX_WAVE_AMPLITUDE = initial_states.MAX_AMPLITUDE
C = initial_states.C
STEPS_PER_FRAME = initial_states.STEPS_PER_FRAME

u_prev = u.copy()  # Start with zero initial velocity


def propagate_t_1D(u):
  global u_prev
  u_next = u.copy()

  for i in range(1, u.shape[0] - 1):
    u_curr_prev = u_prev[i]

    u_next[i] = 2 * u[i] - u_curr_prev + C**2 * (u[i + 1] - 2 * u[i] + u[i - 1])

  u_prev = u.copy()
  u[:] = u_next
  


def make_segments(y):
  x = np.arange(N)
  points = np.array([x, y]).T.reshape(-1, 1, 2)
  return np.concatenate([points[:-1], points[1:]], axis=1)

def plot_wave():
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(np.arange(N), u, color='lightgray', linewidth=1, linestyle='--', label='initial')

  norm = Normalize(vmin=0, vmax=MAX_WAVE_AMPLITUDE)
  lc = LineCollection(make_segments(u), cmap='plasma', norm=norm, linewidth=2)
  lc.set_array(np.abs(u[:-1]))
  ax.add_collection(lc)

  ax.set_ylim(-MAX_WAVE_AMPLITUDE * 1.2, MAX_WAVE_AMPLITUDE * 1.2)
  ax.set_xlabel("Position")
  ax.set_ylabel("Amplitude")
  ax.set_title("1D Wave Equation")
  ax.grid(True)

  plt.pause(1)

  for _ in range(200):
    for _ in range(STEPS_PER_FRAME):
      propagate_t_1D(u)

    lc.set_segments(make_segments(u))
    lc.set_array(np.abs(u[:-1]))
    fig.canvas.draw()
    plt.pause(0.05)

  plt.show()


plot_wave()