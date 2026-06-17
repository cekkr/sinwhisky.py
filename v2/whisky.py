"""
SinWhisky — Python reference implementation.

This module follows the white paper "SinWhisky e FLT" (Riccardo Cecchini, 2014).

It contains two things:

1. The **SinWhisky resampler** (`sinwhisky_resample`) — the algorithm the paper
   actually describes: for every interior sample build the unique circle through
   it and its two neighbours, then synthesise the in-between samples by blending
   the nearby circles with a sine window that favours each circle's centre.
   This is the canonical Python reference and is kept bit-for-bit consistent with
   the C implementation in ``sinwhisky_c/`` so the two can be cross-validated.

2. Circle-fitting / circle-approximation utilities used by the experimental
   circle-based compressor (``find_circle_from_three_points``,
   ``find_optimal_circles``, ``reconstruct_signal``, ...).

``matplotlib`` is imported lazily (only when plotting is requested) so the module
stays importable in headless / minimal environments.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Circle geometry
# ---------------------------------------------------------------------------


def find_circle_from_three_points(p1, p2, p3):
    """
    Calcola il centro e il raggio della circonferenza passante per tre punti.

    Args:
        p1, p2, p3: tuple (x, y) rappresentanti le coordinate dei tre punti

    Returns:
        tuple (center_x, center_y, radius)

    Raises:
        ValueError: se i tre punti sono (quasi) collineari.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Verifica che i punti non siano collineari (doppio dell'area con segno).
    if abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) < 1e-10:
        raise ValueError("I tre punti sono collineari, impossibile trovare una circonferenza unica")

    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x2), 2 * (y3 - y2)]
    ], dtype=float)

    b = np.array([
        x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2,
        x3 ** 2 - x2 ** 2 + y3 ** 2 - y2 ** 2
    ], dtype=float)

    try:
        center_x, center_y = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("Sistema non risolvibile, i punti potrebbero essere quasi collineari")

    radius = float(np.hypot(center_x - x1, center_y - y1))
    return float(center_x), float(center_y), radius


def determine_arc_type(p1, p2, p3, center_x, center_y):
    """
    Determina se l'arco passa sopra o sotto il centro nel punto centrale p2.

    Returns:
        bool: True se il punto centrale sta sopra (o sull') il centro.
    """
    _, y2 = p2
    # L'arco che interpola i tre punti è quello che passa per p2: il ramo è
    # superiore se y2 >= center_y, esattamente come nella versione C.
    return y2 >= center_y


# ---------------------------------------------------------------------------
# SinWhisky resampler (the algorithm described in the paper)
# ---------------------------------------------------------------------------

_PI = np.pi


class _Circle:
    """Circonferenza locale costruita attorno al campione i (coord. scalate)."""

    __slots__ = ("cx", "cy", "r", "aperture", "upper", "valid")

    def __init__(self):
        self.cx = 0.0
        self.cy = 0.0
        self.r = 0.0
        self.aperture = 1.0
        self.upper = True
        self.valid = False


def _fit_circle_local(y0, y1, y2, aperture, collinear_eps):
    """
    Risolve il sistema 2x2 per la circonferenza passante per
    (-aperture, y0), (0, y1), (+aperture, y2).

    Usa esattamente lo stesso determinante della versione C in modo che le due
    implementazioni accettino/rifiutino gli stessi triangoli quasi-collineari.
    """
    c = _Circle()
    c.aperture = aperture

    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
        return c

    x0, x1, x2 = -aperture, 0.0, aperture

    a00 = 2.0 * (x1 - x0)
    a01 = 2.0 * (y1 - y0)
    a10 = 2.0 * (x2 - x1)
    a11 = 2.0 * (y2 - y1)

    b0 = (x1 * x1 - x0 * x0) + (y1 * y1 - y0 * y0)
    b1 = (x2 * x2 - x1 * x1) + (y2 * y2 - y1 * y1)

    det = a00 * a11 - a01 * a10
    if not np.isfinite(det) or abs(det) < collinear_eps:
        return c

    cx = (b0 * a11 - a01 * b1) / det
    cy = (a00 * b1 - b0 * a10) / det
    r = float(np.hypot(cx - x1, cy - y1))

    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r)) or r <= 0.0:
        return c

    c.cx = cx
    c.cy = cy
    c.r = r
    c.upper = (y1 >= cy)
    c.valid = True
    return c


def _circle_eval(c, x_local):
    """Valuta il ramo scelto della circonferenza; None se fuori dall'estensione."""
    dx = x_local - c.cx
    inside = c.r * c.r - dx * dx
    if inside < 0.0:
        return None
    dy = np.sqrt(inside)
    y = c.cy + dy if c.upper else c.cy - dy
    return y if np.isfinite(y) else None


def _window(u, neighbors):
    """Finestra coseno: 1 al centro (u=0), 0 ai bordi (|u|=neighbors)."""
    n = float(neighbors)
    if u <= -n or u >= n:
        return 0.0
    return np.cos(_PI * u / (2.0 * n))


def sinwhisky_resample(samples, zoom,
                       neighbors=1,
                       use_aperture=True,
                       aperture_gain=1.0,
                       aperture_min=1.0,
                       aperture_max=12.0,
                       collinear_eps=1e-10):
    """
    Ricampiona/upsampla un segnale 1D con SinWhisky.

    Per ogni campione interno i si costruisce la circonferenza passante per
    (i-1, i, i+1); i campioni intermedi vengono sintetizzati come media pesata
    (finestra a coseno) delle circonferenze vicine, con fallback lineare quando
    nessuna circonferenza è valida.

    Args:
        samples: sequenza 1D di campioni.
        zoom: numero di campioni da inserire fra due originali (>= 0).
        neighbors: quante circonferenze per lato possono influenzare un campione
            (>= 1). 1 = comportamento classico (le due circonferenze che passano
            con certezza per il punto). Valori maggiori estendono le
            circonferenze oltre i tre punti circoscritti (paper, pag. 7).
        use_aperture: abilita la scalatura dell'apertura sull'asse x.
        aperture_gain/min/max: parametri della formula dell'apertura.
        collinear_eps: soglia sul determinante per punti quasi collineari.

    Returns:
        numpy.ndarray (float64) di lunghezza ``(n-1)*(zoom+1)+1``.
        I campioni originali sono preservati esattamente.
    """
    # Riferimento in piena doppia precisione. La versione C lavora su `float`
    # (32 bit) internamente, quindi nel confronto incrociato C/Python si usa una
    # tolleranza che assorbe l'arrotondamento a 32 bit.
    y = np.asarray(samples, dtype=np.float64)
    n = y.shape[0]

    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return y.copy()

    zoom = int(zoom)
    if zoom < 0:
        zoom = 0
    neighbors = int(neighbors)
    if neighbors < 1:
        neighbors = 1

    n_out = (n - 1) * (zoom + 1) + 1
    out = np.empty(n_out, dtype=np.float64)

    # Precalcola le circonferenze per i centri 1 .. n-2.
    circles = [None] * n
    for i in range(1, n - 1):
        if use_aperture:
            d01 = abs(y[i - 1] - y[i])
            d12 = abs(y[i] - y[i + 1])
            d02 = abs(y[i - 1] - y[i + 1])
            aperture = max(d01, d12, d02) * aperture_gain
            aperture = min(max(aperture, aperture_min), aperture_max)
        else:
            aperture = 1.0
        circles[i] = _fit_circle_local(y[i - 1], y[i], y[i + 1], aperture, collinear_eps)

    nf = float(neighbors)
    w = 0
    for i in range(n - 1):
        out[w] = y[i]
        w += 1

        for k in range(1, zoom + 1):
            t = k / (zoom + 1)
            g = i + t

            jlo = max(1, i - (neighbors - 1))
            jhi = min(n - 2, i + neighbors)

            sum_w = 0.0
            sum_y = 0.0
            for j in range(jlo, jhi + 1):
                c = circles[j]
                if c is None or not c.valid:
                    continue
                u = g - j
                if u <= -nf or u >= nf:
                    continue
                y_pred = _circle_eval(c, u * c.aperture)
                if y_pred is None:
                    continue
                ww = _window(u, neighbors)
                if ww <= 0.0:
                    continue
                sum_w += ww
                sum_y += ww * y_pred

            if sum_w > 1e-20:
                out[w] = sum_y / sum_w
            else:
                out[w] = (1.0 - t) * y[i] + t * y[i + 1]
            w += 1

    out[w] = y[n - 1]
    return out


# ---------------------------------------------------------------------------
# Experimental circle-based compressor (secondary; not part of the paper core)
# ---------------------------------------------------------------------------


def circle_approximation_error(points, center_x, center_y, radius):
    """Errore medio assoluto fra i punti e la circonferenza (|distanza - raggio|)."""
    pts = np.asarray(points, dtype=float)
    dist = np.hypot(pts[:, 0] - center_x, pts[:, 1] - center_y)
    return float(np.mean(np.abs(dist - radius)))


def find_optimal_circles(signal, max_error_threshold=0.01, min_points_per_circle=5):
    """
    Trova una sequenza di circonferenze che approssimano il segnale.

    Returns:
        list di tuple (center_x, center_y, radius, is_upper_arc, start_idx, end_idx)
    """
    signal = [tuple(p) for p in signal]
    circles = []
    n = len(signal)
    start_idx = 0

    while start_idx < n - min_points_per_circle:
        p1 = signal[start_idx]
        p2 = signal[start_idx + 1]
        p3 = signal[start_idx + 2]

        try:
            center_x, center_y, radius = find_circle_from_three_points(p1, p2, p3)
            is_upper_arc = determine_arc_type(p1, p2, p3, center_x, center_y)
        except ValueError:
            start_idx += 1
            continue

        end_idx = start_idx + 3
        current_points = list(signal[start_idx:end_idx])

        while end_idx < n:
            test_points = current_points + [signal[end_idx]]
            error = circle_approximation_error(test_points, center_x, center_y, radius)
            if error <= max_error_threshold:
                current_points = test_points
                end_idx += 1
            else:
                break

        circles.append((center_x, center_y, radius, is_upper_arc, start_idx, end_idx - 1))

        # Avanza garantendo progresso (>= 1) anche con circonferenze minime.
        next_start = end_idx - 2
        if next_start <= start_idx:
            next_start = start_idx + 1
        start_idx = next_start

    return circles


def reconstruct_signal(circles, x_values):
    """
    Ricostruisce il segnale dai parametri delle circonferenze.

    A differenza della versione originale non usa confronti di uguaglianza fra
    float (fragili) ed è vettorizzata: per ogni circonferenza assegna i valori y
    sugli indici coperti dal suo intervallo [start_idx, end_idx].
    """
    x_values = np.asarray(x_values, dtype=float)
    y_reconstructed = np.zeros_like(x_values)

    for center_x, center_y, radius, is_upper_arc, start_idx, end_idx in circles:
        lo = x_values[start_idx]
        hi = x_values[end_idx]
        mask = (x_values >= lo) & (x_values <= hi)
        dx = x_values[mask] - center_x
        inside = radius ** 2 - dx ** 2
        valid = inside >= 0.0
        dy = np.zeros_like(dx)
        dy[valid] = np.sqrt(inside[valid])
        y_seg = (center_y + dy) if is_upper_arc else (center_y - dy)
        idx = np.nonzero(mask)[0]
        y_reconstructed[idx[valid]] = y_seg[valid]

    return y_reconstructed


def encode_circles_for_compression(circles):
    """
    Codifica compatta dei parametri (delta-encoding del centro x).

    A differenza della prima bozza conserva tutto ciò che serve a ``decode`` per
    una ricostruzione fedele: (delta_center_x, center_y, radius, is_upper_arc,
    start_idx, end_idx).
    """
    encoded = []
    prev_x = 0.0
    for center_x, center_y, radius, is_upper_arc, start_idx, end_idx in circles:
        encoded.append((center_x - prev_x, center_y, radius, bool(is_upper_arc), start_idx, end_idx))
        prev_x = center_x
    return encoded


def decode_circles_from_compression(encoded_data, x_values):
    """Decodifica e ricostruisce il segnale (inverso di ``encode``+``reconstruct``)."""
    circles = []
    current_x = 0.0
    for delta_x, center_y, radius, is_upper_arc, start_idx, end_idx in encoded_data:
        current_x += delta_x
        circles.append((current_x, center_y, radius, is_upper_arc, start_idx, end_idx))
    return reconstruct_signal(circles, x_values)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def example_resample():
    """Mostra SinWhisky che 'zooma' una sinusoide a basso sample-rate."""
    x = np.linspace(0, 2 * np.pi, 9)
    y = np.sin(x)
    up = sinwhisky_resample(y, zoom=5)
    print("input :", np.round(y, 3))
    print("output:", len(up), "campioni (zoom=5)")
    return up


def example_compression(show=False):
    x = np.linspace(0, 4 * np.pi, 1000)
    y = np.sin(x)
    signal = list(zip(x, y))

    circles = find_optimal_circles(signal, max_error_threshold=0.01)
    encoded = encode_circles_for_compression(circles)

    original_size = len(signal) * 2
    compressed_size = len(encoded) * 3
    compression_ratio = original_size / max(compressed_size, 1)

    y_reconstructed = reconstruct_signal(circles, x)
    mse = float(np.mean((y - y_reconstructed) ** 2))

    print(f"Numero di punti originali: {len(signal)}")
    print(f"Numero di circonferenze: {len(circles)}")
    print(f"Rapporto di compressione: {compression_ratio:.2f}x")
    print(f"Errore quadratico medio: {mse:.6f}")

    if show:
        import matplotlib.pyplot as plt  # import pigro: solo se serve disegnare
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'b-', label='Segnale originale')
        plt.plot(x, y_reconstructed, 'r--', label='Segnale ricostruito')
        plt.title(f'Compressione circolare (Rapporto: {compression_ratio:.2f}x)')
        plt.xlabel('Tempo')
        plt.ylabel('Ampiezza')
        plt.legend()
        plt.grid(True)
        plt.show()

    return circles, y_reconstructed


if __name__ == "__main__":
    example_resample()
    example_compression(show=False)
