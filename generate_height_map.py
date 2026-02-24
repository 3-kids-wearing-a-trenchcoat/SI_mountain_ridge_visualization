from __future__ import annotations
import numpy as np

def gaussian_random_field(shape, beta=2.0, seed=None):
    """
    Spectral method: produce a Gaussian random field with power-spectrum ~ 1/k^beta.
    - shape: (nx, ny)
    - beta: bigger => smoother. beta ~ 0..4 works; 2 is a nice midrange.
    """
    rng = np.random.RandomState(seed)
    nx, ny = shape
    # frequencies (normalized)
    kx = np.fft.fftfreq(nx)[:, None]
    ky = np.fft.fftfreq(ny)[None, :]
    k = np.sqrt(kx*kx + ky*ky)
    k[0, 0] = 1.0  # avoid divide by zero
    # amplitude scaling
    amplitude = 1.0 / (k ** (beta/2.0))
    # complex Gaussian coefficients
    noise = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    spectrum = amplitude * noise
    spectrum[0, 0] = 0.0  # zero mean
    field = np.fft.ifft2(spectrum).real
    # standardize
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field

def fractal_octaves(shape, octaves=6, persistence=0.5, seed=None):
    """
    Simple multi-scale (value-noise-like) fractal: sum of coarse random fields upsampled.
    Uses repeated tiling/interpolation via np.repeat for simplicity.
    """
    rng = np.random.RandomState(seed)
    nx, ny = shape
    field = np.zeros(shape, dtype=float)
    amplitude = 1.0
    total_amp = 0.0
    for o in range(octaves):
        freq = 2 ** o
        # coarse grid size (keep at least 2x2)
        cx = max(2, nx // freq)
        cy = max(2, ny // freq)
        coarse = rng.normal(size=(cx, cy))
        # upsample by repeating (blocky but okay for many octaves)
        up = np.repeat(np.repeat(coarse, nx//cx + 1, axis=0)[:nx, :], ny//cy + 1, axis=1)[:nx, :ny]
        field += amplitude * up
        total_amp += amplitude
        amplitude *= persistence
    field /= (total_amp + 1e-12)
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field

def many_gaussian_bumps(shape, n_bumps=800, min_sigma=1.0, max_sigma=30.0, seed=None, amplitude_range=(-1.5, 1.5)):
    """
    Add many random Gaussian bumps/pits. Useful to create lots of local minima/maxima.
    amplitude_range: negative -> pits, positive -> bumps. You can mix.
    """
    rng = np.random.RandomState(seed)
    nx, ny = shape
    xs = np.linspace(0, nx-1, nx)
    ys = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    field = np.zeros(shape, dtype=float)
    for _ in range(n_bumps):
        cx = rng.uniform(0, nx)
        cy = rng.uniform(0, ny)
        sigma = rng.uniform(min_sigma, max_sigma)
        amp = rng.uniform(amplitude_range[0], amplitude_range[1])
        d2 = (X - cx)**2 + (Y - cy)**2
        field += amp * np.exp(-0.5 * d2 / (sigma*sigma))
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field

def combine_landscape(shape, seed=None):
    """Combine the techniques above into one complex landscape, then normalize."""
    # base spectral field (gives correlated randomness)
    grf = gaussian_random_field(shape, beta=2.2, seed=(seed or 0) + 1)
    # fractal detail (adds multi-scale roughness)
    fract = fractal_octaves(shape, octaves=6, persistence=0.5, seed=(seed or 0) + 2)
    # many small gaussians (creates many local minima/pits)
    bumps = many_gaussian_bumps(shape, n_bumps=400, min_sigma=1.0, max_sigma=18.0, seed=(seed or 0) + 3,
                                amplitude_range=(-1.0, 1.0))
    # sinusoidal ridges to add long-range structure
    nx, ny = shape
    xs = np.linspace(0, 2*np.pi*3, nx)  # 3 wavelengths across x
    ys = np.linspace(0, 2*np.pi*5, ny)  # 5 wavelengths across y
    Sx, Sy = np.meshgrid(xs, ys, indexing='ij')
    ridges = 0.5 * (np.sin(Sx*1.5) * np.cos(Sy*0.9))
    # combine with weights
    field = 0.45*grf + 0.25*fract + 0.45*bumps + 0.25*ridges
    # add a clear global minimum (a large deep pit) so the global optimum exists
    # choose a random center or fixed center
    rng = np.random.RandomState(seed)
    cx = int(nx*0.7)  # you can randomize: rng.randint(nx)
    cy = int(ny*0.3)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    global_pit = -6.0 * np.exp(-0.5 * ((X-cx)**2 + (Y-cy)**2) / ( (min(nx, ny)/8.0)**2 ))
    field += global_pit
    # final normalization to control absolute scale
    field = (field - field.mean()) / (field.std() + 1e-12)

    # shift values so that lowest is 0
    # min_height = np.min(field)
    # if min_height < 0:
    #     field -= min_height

    return field