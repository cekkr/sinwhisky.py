import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def find_circle_from_three_points(p1, p2, p3):
    """
    Calcola il centro e il raggio della circonferenza passante per tre punti.

    Args:
        p1, p2, p3: tuple (x, y) rappresentanti le coordinate dei tre punti

    Returns:
        tuple (center_x, center_y, radius)
    """
    # Coordinate dei punti
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Verifica che i punti non siano collineari
    if abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) < 1e-10:
        raise ValueError("I tre punti sono collineari, impossibile trovare una circonferenza unica")

    # Calcolo dei parametri della circonferenza
    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x2), 2 * (y3 - y2)]
    ])

    b = np.array([
        x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2,
        x3 ** 2 - x2 ** 2 + y3 ** 2 - y2 ** 2
    ])

    # Risoluzione del sistema per trovare il centro
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("Sistema non risolvibile, i punti potrebbero essere quasi collineari")

    # Estrazione delle coordinate del centro
    center_x, center_y = center

    # Calcolo del raggio
    radius = np.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

    return center_x, center_y, radius


def determine_arc_type(p1, p2, p3, center_x, center_y):
    """
    Determina se l'arco è superiore o inferiore rispetto al centro.

    Returns:
        bool: True per arco superiore, False per arco inferiore
    """
    # Calcolo dell'angolo centrale per il punto di mezzo
    x2, y2 = p2
    dx = x2 - center_x
    dy = y2 - center_y

    # Determina se l'arco è nella parte superiore o inferiore
    # considerando la posizione relativa del punto centrale
    angle = np.arctan2(dy, dx)

    # Se l'angolo è nel primo o quarto quadrante,
    # l'arco è nella parte superiore
    is_upper_arc = (-np.pi / 2 <= angle <= np.pi / 2)

    return is_upper_arc


def circle_approximation_error(points, center_x, center_y, radius):
    """
    Calcola l'errore di approssimazione tra i punti e la circonferenza.

    Args:
        points: lista di tuple (x, y)
        center_x, center_y: coordinate del centro
        radius: raggio della circonferenza

    Returns:
        float: errore quadratico medio
    """
    errors = []
    for x, y in points:
        # Distanza dal centro
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Errore rispetto al raggio
        error = abs(distance - radius)
        errors.append(error)

    return np.mean(errors)


def find_optimal_circles(signal, max_error_threshold=0.01, min_points_per_circle=5):
    """
    Trova la sequenza ottimale di circonferenze che approssimano il segnale.

    Args:
        signal: array di tuple (x, y) rappresentanti il segnale
        max_error_threshold: errore massimo accettabile
        min_points_per_circle: numero minimo di punti per circonferenza

    Returns:
        list: lista di tuple (center_x, center_y, radius, is_upper_arc, start_idx, end_idx)
    """
    circles = []
    start_idx = 0
    n = len(signal)

    while start_idx < n - min_points_per_circle:
        # Inizia con i primi tre punti
        p1 = signal[start_idx]
        p2 = signal[start_idx + 1]
        p3 = signal[start_idx + 2]

        try:
            center_x, center_y, radius = find_circle_from_three_points(p1, p2, p3)
            is_upper_arc = determine_arc_type(p1, p2, p3, center_x, center_y)
        except ValueError:
            # Se i punti sono collineari, passa ai successivi
            start_idx += 1
            continue

        # Cerca il massimo numero di punti che possono essere approssimati
        # dalla stessa circonferenza entro la soglia di errore
        end_idx = start_idx + 3
        current_points = signal[start_idx:end_idx]

        while end_idx < n:
            # Aggiungi il punto successivo
            next_point = signal[end_idx]
            test_points = current_points + [next_point]

            # Calcola l'errore con il punto aggiuntivo
            error = circle_approximation_error(test_points, center_x, center_y, radius)

            if error <= max_error_threshold:
                # Il punto può essere approssimato dalla stessa circonferenza
                current_points = test_points
                end_idx += 1
            else:
                # Il punto richiede una nuova circonferenza
                break

        # Memorizza i parametri della circonferenza
        circles.append((center_x, center_y, radius, is_upper_arc, start_idx, end_idx - 1))

        # Passa al prossimo gruppo di punti
        start_idx = end_idx - 2  # Sovrapposizione di 2 punti per continuità

    return circles


def reconstruct_signal(circles, x_values):
    """
    Ricostruisce il segnale dai parametri delle circonferenze.

    Args:
        circles: lista di tuple (center_x, center_y, radius, is_upper_arc, start_idx, end_idx)
        x_values: array di valori x per cui ricostruire il segnale

    Returns:
        array: valori y ricostruiti
    """
    y_reconstructed = np.zeros_like(x_values)

    for center_x, center_y, radius, is_upper_arc, start_idx, end_idx in circles:
        # Trova i valori x che rientrano in questa circonferenza
        mask = (x_values >= x_values[start_idx]) & (x_values <= x_values[end_idx])
        x_segment = x_values[mask]

        # Calcola i valori y corrispondenti sulla circonferenza
        for i, x in enumerate(x_segment):
            # Calcola la coordinata y sulla circonferenza
            # per il dato valore x
            dx = x - center_x

            # Assicurati che il punto sia nel dominio della circonferenza
            if abs(dx) > radius:
                continue

            # Calcola la coordinata y (ci sono due possibili valori)
            dy = np.sqrt(radius ** 2 - dx ** 2)

            # Scegli l'arco superiore o inferiore
            if is_upper_arc:
                y = center_y + dy
            else:
                y = center_y - dy

            # Assegna il valore ricostruito
            idx = np.where(x_values == x)[0][0]
            y_reconstructed[idx] = y

    return y_reconstructed


def encode_circles_for_compression(circles, original_signal):
    """
    Codifica i parametri delle circonferenze in un formato compatto per la compressione.

    Args:
        circles: lista di tuple (center_x, center_y, radius, is_upper_arc, start_idx, end_idx)
        original_signal: array di tuple (x, y) del segnale originale

    Returns:
        list: lista di tuple (delta_x, radius, is_upper_arc)
    """
    encoded_data = []
    prev_x = 0  # Posizione x iniziale

    for center_x, center_y, radius, is_upper_arc, start_idx, end_idx in circles:
        # Calcola lo spostamento rispetto alla posizione precedente
        delta_x = center_x - prev_x

        # Aggiorna la posizione precedente
        prev_x = center_x

        # Memorizza solo i parametri essenziali
        encoded_data.append((delta_x, radius, is_upper_arc))

    return encoded_data


def decode_circles_from_compression(encoded_data, x_values):
    """
    Decodifica i parametri delle circonferenze dal formato compresso.

    Args:
        encoded_data: lista di tuple (delta_x, radius, is_upper_arc)
        x_values: array di valori x per cui ricostruire il segnale

    Returns:
        array: valori y ricostruiti
    """
    y_reconstructed = np.zeros_like(x_values)
    current_x = 0

    for delta_x, radius, is_upper_arc in encoded_data:
        # Calcola la posizione x del centro
        center_x = current_x + delta_x
        current_x = center_x

        # Determina l'intervallo di valori x interessati da questa circonferenza
        # Questa è una semplificazione: nella realtà, dovremmo memorizzare
        # anche gli indici di inizio e fine per ogni circonferenza
        x_min = center_x - radius
        x_max = center_x + radius

        # Trova i valori x che rientrano in questa circonferenza
        mask = (x_values >= x_min) & (x_values <= x_max)
        x_segment = x_values[mask]

        # Calcola i valori y corrispondenti sulla circonferenza
        for i, x in enumerate(x_segment):
            # Calcola la coordinata y sulla circonferenza
            dx = x - center_x

            # Assicurati che il punto sia nel dominio della circonferenza
            if abs(dx) > radius:
                continue

            # Calcola la coordinata y (ci sono due possibili valori)
            dy = np.sqrt(radius ** 2 - dx ** 2)

            # Scegli l'arco superiore o inferiore
            if is_upper_arc:
                y = dy  # Centra la circonferenza sull'asse x per semplicità
            else:
                y = -dy

            # Assegna il valore ricostruito
            idx = np.where(x_values == x)[0][0]
            y_reconstructed[idx] = y

    return y_reconstructed


# Esempio di utilizzo
def example_compression():
    # Generazione di un segnale di esempio (sinusoide)
    x = np.linspace(0, 4 * np.pi, 1000)
    y = np.sin(x)
    signal = list(zip(x, y))

    # Trova le circonferenze ottimali
    circles = find_optimal_circles(signal, max_error_threshold=0.01)

    # Codifica i parametri delle circonferenze per la compressione
    encoded_data = encode_circles_for_compression(circles, signal)

    # Calcola il rapporto di compressione
    original_size = len(signal) * 2  # Due float per ogni punto
    compressed_size = len(encoded_data) * 3  # Tre valori per ogni circonferenza
    compression_ratio = original_size / compressed_size

    print(f"Numero di punti originali: {len(signal)}")
    print(f"Numero di circonferenze: {len(circles)}")
    print(f"Rapporto di compressione: {compression_ratio:.2f}x")

    # Ricostruisci il segnale
    y_reconstructed = reconstruct_signal(circles, x)

    # Calcola l'errore di ricostruzione
    mse = np.mean((y - y_reconstructed) ** 2)
    print(f"Errore quadratico medio: {mse:.6f}")

    # Visualizza i risultati
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', label='Segnale originale')
    plt.plot(x, y_reconstructed, 'r--', label='Segnale ricostruito')

    # Visualizza i centri delle circonferenze
    for center_x, center_y, radius, is_upper_arc, _, _ in circles:
        plt.plot(center_x, center_y, 'go')
        # Disegna un cerchio parziale
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = center_x + radius * np.cos(theta)
        circle_y = center_y + radius * np.sin(theta)
        plt.plot(circle_x, circle_y, 'g:', alpha=0.3)

    plt.title(f'Compressione audio con approssimazione circolare (Rapporto: {compression_ratio:.2f}x)')
    plt.xlabel('Tempo')
    plt.ylabel('Ampiezza')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    example_compression()