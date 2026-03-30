import numpy as np

def get_square_pulse(length, amplitude):
  pulse = np.zeros(length)
  pulse[int(length * 0.4): int(length * 0.6)] = amplitude

  return pulse

def get_gaussian_pulse(shape, amplitude, sigma=10, center=None):
  if isinstance(shape, int):
    shape = (shape,)

  dims = len(shape)

  if center is None:
    center = [s // 2 for s in shape]
  elif isinstance(center, (int, float)):
    center = [center]
  if isinstance(sigma, (int, float)):
    sigma = [sigma] * dims

  axes = [np.arange(s) for s in shape]
  grids = np.meshgrid(*axes, indexing='ij')

  exponent = sum(((g - c) / sig) ** 2 for g, c, sig in zip(grids, center, sigma))
  return amplitude * np.exp(-0.5 * exponent)



if __name__ == "__main__":
  import matplotlib.pyplot as plt

  # 1D
  pulse_1d = get_gaussian_pulse(200, 1.0, sigma=10)
  plt.plot(pulse_1d)
  plt.title("Gaussian Pulse 1D")
  plt.xlabel("Position")
  plt.ylabel("Amplitude")
  plt.grid()
  plt.show()

  # 2D
  pulse_2d = get_gaussian_pulse((100, 100), 1.0, sigma=10)
  plt.imshow(pulse_2d, origin='lower')
  plt.colorbar()
  plt.title("Gaussian Pulse 2D")
  plt.show()

  