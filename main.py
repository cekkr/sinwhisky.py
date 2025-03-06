import numpy as np
from scipy.io import wavfile
import struct
import math
import os
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings("ignore")

class CircleParameters:
    """Classe per rappresentare i parametri di un cerchio."""
    def __init__(self, 
                 radius: float, 
                 x_offset: float = 0.0, 
                 use_upper_half: bool = True,
                 amplitude: float = 1.0):
        """
        Inizializza i parametri del cerchio.
        
        Args:
            radius: raggio del cerchio
            x_offset: offset orizzontale rispetto al cerchio precedente
            use_upper_half: True se usare la metà superiore, False se usare la metà inferiore
            amplitude: ampiezza del cerchio (fattore di scala)
        """
        self.radius = radius
        self.x_offset = x_offset
        self.use_upper_half = use_upper_half
        self.amplitude = amplitude
    
    def to_bytes(self, precision: str = 'float16') -> bytes:
        """Converte i parametri del cerchio in bytes per la serializzazione."""
        if precision == 'float16':
            # dtype = np.float16  # Not needed for struct.pack
            pack_format = 'e'  # formato per float16
        elif precision == 'float32':
            # dtype = np.float32  # Not needed for struct.pack
            pack_format = 'f'  # formato per float32
        else:
            raise ValueError(f"Precisione non supportata: {precision}")
        
        # Normalizza i valori per massimizzare la precisione dei float16
        max_radius = 10.0  # Valore massimo atteso per il raggio
        max_offset = 1000.0  # Valore massimo atteso per l'offset
        max_amplitude = 1.0  # Valore massimo atteso per l'ampiezza
        
        # Calcola fattori di normalizzazione per rimanere nel range ottimale di float16
        normalized_radius = min(self.radius / max_radius, 1.0)
        normalized_offset = min(self.x_offset / max_offset, 1.0)
        normalized_amplitude = min(self.amplitude / max_amplitude, 1.0)
        
        # Converte parametri normalizzati in bytes
        result = struct.pack(pack_format, normalized_radius)
        result += struct.pack(pack_format, normalized_offset)
        result += struct.pack(pack_format, normalized_amplitude)
        
        # Usa un singolo bit per il flag upper/lower half
        flag_byte = 1 if self.use_upper_half else 0
        result += struct.pack('B', flag_byte)
        
        return result
    
    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0, precision: str = 'float16') -> Tuple['CircleParameters', int]:
        """
        Crea un'istanza di CircleParameters da bytes.
        
        Args:
            data: bytes da cui leggere i parametri
            offset: offset da cui iniziare a leggere
            precision: precisione dei float ('float16' o 'float32')
            
        Returns:
            Una tupla (istanza CircleParameters, nuovo offset)
        """
        if precision == 'float16':
            float_size = 2  # bytes
            pack_format = 'e'
            max_radius = 10.0
            max_offset = 1000.0
            max_amplitude = 1.0
        elif precision == 'float32':
            float_size = 4  # bytes
            pack_format = 'f'
            max_radius = 10.0
            max_offset = 1000.0
            max_amplitude = 1.0
        else:
            raise ValueError(f"Precisione non supportata: {precision}")
        
        try:
            # Legge e denormalizza il raggio
            norm_radius = struct.unpack(pack_format, data[offset:offset+float_size])[0]
            radius = norm_radius * max_radius
            offset += float_size
            
            # Legge e denormalizza l'offset orizzontale
            norm_offset = struct.unpack(pack_format, data[offset:offset+float_size])[0]
            x_offset = norm_offset * max_offset
            offset += float_size
            
            # Legge e denormalizza l'ampiezza
            norm_amplitude = struct.unpack(pack_format, data[offset:offset+float_size])[0]
            amplitude = norm_amplitude * max_amplitude
            offset += float_size
            
            # Legge il flag per la metà del cerchio (come byte singolo)
            flag_byte = struct.unpack('B', data[offset:offset+1])[0]
            use_upper_half = flag_byte > 0
            offset += 1
            
            # Verifica validità dei parametri
            if not (0 <= radius <= max_radius * 1.1):
                radius = 1.0  # valore sicuro di fallback
            
            if not (0 <= x_offset <= max_offset * 1.1):
                x_offset = 1.0  # valore sicuro di fallback
                
            if not (0 <= amplitude <= max_amplitude * 1.1):
                amplitude = 0.5  # valore sicuro di fallback
            
            return cls(radius, x_offset, use_upper_half, amplitude), offset
            
        except struct.error:
            # Gestisce errori di unpacking con valori predefiniti sicuri
            print("Errore nella lettura dei dati del cerchio. Usando valori predefiniti.")
            return cls(1.0, 1.0, True, 0.5), offset + float_size*3 + 1


class CircularAudioCompressor:
    """Classe per la compressione e decompressione audio usando approssimazione a cerchi."""
    
    def __init__(self, precision: str = 'float16', min_accuracy: float = 0.1, 
                 segments_per_thread: int = 10, use_threading: bool = True):
        """
        Inizializza il compressore.
        
        Args:
            precision: precisione dei float ('float16' o 'float32')
            min_accuracy: precisione minima dell'approssimazione
            segments_per_thread: numero di segmenti da processare per thread
            use_threading: se utilizzare il multithreading
        """
        self.precision = precision
        self.min_accuracy = min_accuracy
        self.segments_per_thread = segments_per_thread
        self.use_threading = use_threading
        
        if precision == 'float16':
            self.dtype = np.float16
            # self.max_value = np.finfo(np.float16).max # Not used
        elif precision == 'float32':
            self.dtype = np.float32
            # self.max_value = np.finfo(np.float32).max # Not used
        else:
            raise ValueError(f"Precisione non supportata: {precision}")
    
    def _process_segment(self, segment, debug=False):
        """
        Processa un singolo segmento audio per la compressione.
        Utilizzato sia direttamente che nei thread.
        
        Args:
            segment: segmento audio da processare
            debug: se True, stampa informazioni di debug
            
        Returns:
            Lista di CircleParameters che approssimano il segmento
        """
        # Approssima il segmento con cerchi
        circles = self._approximate_with_circles(segment)
        
        if debug and len(circles) > 0:
            print(f"Segmento approssimato con {len(circles)} cerchi")
            
        return circles
        
    def compress(self, input_file: str, output_file: str, debug=False) -> None:
        """
        Comprime un file audio WAV usando l'approssimazione a cerchi.
        
        Args:
            input_file: percorso del file WAV di input
            output_file: percorso del file compresso di output
            debug: se True, stampa informazioni di debug
        """
        print("Avvio compressione...")
        
        # Legge il file audio
        try:
            sample_rate, audio_data = wavfile.read(input_file)
        except Exception as e:
            print(f"Errore nella lettura del file audio: {e}")
            return
            
        # Assicura che i dati siano mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalizza l'audio tra -1 e 1
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 0:
            audio_data /= max_abs
        
        # Applica un filtro per rimuovere rumore ad alta frequenza
        from scipy import signal
        # Filtro passa-basso per rimuovere frequenze indesiderate
        b, a = signal.butter(4, 0.3, 'low')
        audio_data = signal.filtfilt(b, a, audio_data)
        
        # Divide l'audio in segmenti
        segment_length = 500  # Lunghezza ridotta per migliore approssimazione
        num_segments = math.ceil(len(audio_data) / segment_length)
        
        if debug:
            print(f"File audio: {input_file}")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"Lunghezza: {len(audio_data)} campioni ({len(audio_data)/sample_rate:.2f} secondi)")
            print(f"Numero di segmenti: {num_segments} (lunghezza segmento: {segment_length})")
        
        # Prepara l'header del file compresso
        header = struct.pack('I', sample_rate)  # Sample rate
        header += struct.pack('I', len(audio_data))  # Lunghezza originale in campioni
        header += struct.pack('I', segment_length)  # Lunghezza del segmento
        header += struct.pack('I', num_segments)  # Numero di segmenti
        header += struct.pack('f', max_abs)  # Fattore di normalizzazione
        header += struct.pack('B', 1 if self.precision == 'float16' else 2)  # Codice precisione
        
        # Comprime ogni segmento
        compressed_data = bytearray(header)
        total_circles = 0
        
        # Implementazione multithreading
        if self.use_threading and num_segments > 1:
            import threading
            from queue import Queue
            
            print("Utilizzo multithreading per la compressione...")
            
            # Coda di risultati
            results_queue = Queue()
            
            # Funzione worker per thread
            def worker(segment_batch, batch_indices):
                batch_results = []
                for i, segment_idx in enumerate(batch_indices):
                    start = segment_idx * segment_length
                    end = min(start + segment_length, len(audio_data))
                    segment = audio_data[start:end]
                    
                    # Approssima il segmento con cerchi
                    circles = self._process_segment(segment, debug)
                    batch_results.append((segment_idx, circles))
                    
                results_queue.put(batch_results)
            
            # Crea batch di segmenti per i thread
            threads = []
            num_threads = min(os.cpu_count() or 4, num_segments)
            segments_per_thread = max(1, num_segments // num_threads)
            
            if debug:
                print(f"Utilizzo {num_threads} thread con {segments_per_thread} segmenti per thread")
            
            # Avvia i thread
            for t in range(num_threads):
                start_idx = t * segments_per_thread
                end_idx = min(start_idx + segments_per_thread, num_segments)
                if start_idx >= end_idx:
                    continue
                    
                batch_indices = list(range(start_idx, end_idx))
                # batch = [audio_data[i * segment_length:min((i + 1) * segment_length, len(audio_data))] for i in batch_indices] # Not used
                
                thread = threading.Thread(target=worker, args=([None], batch_indices))  # Pass dummy batch
                thread.start()
                threads.append(thread)
            
            # Attendi il completamento di tutti i thread
            for thread in threads:
                thread.join()
                
            # Raccogli i risultati
            all_results = []
            while not results_queue.empty():
                all_results.extend(results_queue.get())
                
            # Ordina i risultati per indice di segmento
            all_results.sort(key=lambda x: x[0])
            
            # Crea i dati compressi ordinati
            segment_data_map = {}
            for segment_idx, circles in all_results:
                segment_bytes = struct.pack('H', len(circles))  # Numero di cerchi
                for circle in circles:
                    segment_bytes += circle.to_bytes(self.precision)
                segment_data_map[segment_idx] = segment_bytes
                total_circles += len(circles)
                
            # Aggiungi i dati dei segmenti in ordine
            for i in range(num_segments):
                if i in segment_data_map:
                    compressed_data.extend(segment_data_map[i])
                else:
                    # Segmento vuoto se mancante
                    compressed_data.extend(struct.pack('H', 0))
                    
        else:
            # Versione senza multithreading
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, len(audio_data))
                segment = audio_data[start:end]
                
                # Approssima il segmento con cerchi
                circles = self._approximate_with_circles(segment)
                total_circles += len(circles)
                
                # Serializza i cerchi
                segment_bytes = struct.pack('H', len(circles))  # Numero di cerchi
                for circle in circles:
                    segment_bytes += circle.to_bytes(self.precision)
                
                compressed_data.extend(segment_bytes)
                
                if debug and i % 100 == 0:
                    print(f"Processato segmento {i}/{num_segments}")
        
        # Applica una compressione aggiuntiva per ridurre la dimensione del file
        try:
            import zlib
            compressed_data = zlib.compress(compressed_data, level=9)
            use_zlib = True
            # Aggiungi un flag per indicare l'uso di zlib
            compressed_data = b'ZLIB' + compressed_data
        except ImportError:
            use_zlib = False
            print("Modulo zlib non disponibile, nessuna compressione aggiuntiva applicata")
            
        # Salva il file compresso
        with open(output_file, 'wb') as f:
            f.write(compressed_data)
        
        # Stampa informazioni sulla compressione
        original_size = os.path.getsize(input_file)
        compressed_size = os.path.getsize(output_file)
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        
        print(f"Compressione completata:")
        print(f"Dimensione originale: {original_size:,} bytes")
        print(f"Dimensione compressa: {compressed_size:,} bytes")
        print(f"Rapporto di compressione: {ratio:.2f}x")
        print(f"Numero totale di cerchi: {total_circles}")
        print(f"Media cerchi per segmento: {total_circles/num_segments:.2f}")
        
        if use_zlib:
            print(f"Compressione zlib applicata")
    
    def decompress(self, input_file: str, output_file: str, debug=False) -> None:
        """
        Decomprime un file audio precedentemente compresso.
        
        Args:
            input_file: percorso del file compresso
            output_file: percorso del file WAV decompresso
            debug: se True, stampa informazioni di debug
        """
        print("Avvio decompressione...")
        
        # Legge il file compresso
        with open(input_file, 'rb') as f:
            data = f.read()
            
        # Verifica se è stata applicata la compressione zlib
        if data[:4] == b'ZLIB':
            try:
                import zlib
                data = zlib.decompress(data[4:])
                if debug:
                    print("Decompressione zlib applicata")
            except (ImportError, zlib.error) as e:
                print(f"Errore nella decompressione zlib: {e}")
                return
        
        # Estrae l'header
        offset = 0
        try:
            sample_rate = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            original_length = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            segment_length = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            num_segments = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            normalization_factor = struct.unpack('f', data[offset:offset+4])[0]
            offset += 4
            
            precision_code = struct.unpack('B', data[offset:offset+1])[0]
            precision = 'float16' if precision_code == 1 else 'float32'
            offset += 1
            
            if debug:
                print(f"Header estratto:")
                print(f"Sample rate: {sample_rate} Hz")
                print(f"Lunghezza originale: {original_length} campioni")
                print(f"Lunghezza segmento: {segment_length}")
                print(f"Numero segmenti: {num_segments}")
                print(f"Fattore normalizzazione: {normalization_factor}")
                print(f"Precisione: {precision}")
                
        except struct.error as e:
            print(f"Errore nell'estrazione dell'header: {e}")
            return
            
        # Validazione basilare dei dati estratti
        if (sample_rate <= 0 or original_length <= 0 or 
            segment_length <= 0 or num_segments <= 0 or
            normalization_factor <= 0):
            print("Errore: valori dell'header non validi")
            return
        
        # Ricostruisce l'audio
        reconstructed_audio = np.zeros(original_length, dtype=np.float32)
        
        # Implementazione multithreading per la decompressione
        if self.use_threading and num_segments > 1:
            import threading
            from queue import Queue
            
            if debug:
                print("Utilizzo multithreading per la decompressione...")
            
            # Estrai tutti i dati dei segmenti e i loro parametri
            segment_data = []
            current_offset = offset
            
            for i in range(num_segments):
                try:
                    # Legge il numero di cerchi per questo segmento
                    num_circles = struct.unpack('H', data[current_offset:current_offset+2])[0]
                    current_offset += 2
                    
                    # Estrae i parametri dei cerchi
                    circles = []
                    for _ in range(num_circles):
                        circle, current_offset = CircleParameters.from_bytes(data, current_offset, precision)
                        circles.append(circle)
                        
                    segment_data.append((i, circles))
                    
                except (struct.error, IndexError) as e:
                    print(f"Errore nella lettura del segmento {i}: {e}")
                    segment_data.append((i, []))
            
            # Funzione worker per thread
            def decompression_worker(segment_batch, results_queue):
                batch_results = []
                for segment_idx, circles in segment_batch:
                    start_sample = segment_idx * segment_length
                    end_sample = min(start_sample + segment_length, original_length)
                    
                    # Ricostruisce il segmento audio dai cerchi
                    segment = self._reconstruct_from_circles(circles, end_sample - start_sample)
                    batch_results.append((segment_idx, segment))
                    
                results_queue.put(batch_results)
            
            # Crea batch di segmenti per i thread
            threads = []
            results_queue = Queue()
            num_threads = min(os.cpu_count() or 4, len(segment_data))
            segments_per_thread = max(1, len(segment_data) // num_threads)
            
            if debug:
                print(f"Utilizzo {num_threads} thread con {segments_per_thread} segmenti per thread")
            
            # Avvia i thread
            for t in range(num_threads):
                start_idx = t * segments_per_thread
                end_idx = min(start_idx + segments_per_thread, len(segment_data))
                if start_idx >= end_idx:
                    continue
                    
                batch = segment_data[start_idx:end_idx]
                thread = threading.Thread(target=decompression_worker, args=(batch, results_queue))
                thread.start()
                threads.append(thread)
            
            # Attendi il completamento di tutti i thread
            for thread in threads:
                thread.join()
                
            # Raccogli i risultati
            reconstructed_segments = {}
            while not results_queue.empty():
                batch_results = results_queue.get()
                for segment_idx, segment in batch_results:
                    reconstructed_segments[segment_idx] = segment
            
            # Assembla i segmenti nell'array audio finale
            for segment_idx, segment in sorted(reconstructed_segments.items()):
                start_sample = segment_idx * segment_length
                end_sample = min(start_sample + segment_length, original_length)
                segment_length_actual = end_sample - start_sample
                
                if len(segment) > segment_length_actual:
                    segment = segment[:segment_length_actual]
                elif len(segment) < segment_length_actual:
                    # Gestisci il caso in cui il segmento è più corto del previsto
                    padded_segment = np.zeros(segment_length_actual, dtype=np.float32)
                    padded_segment[:len(segment)] = segment
                    segment = padded_segment
                
                reconstructed_audio[start_sample:end_sample] = segment
                
        else:
            # Versione senza multithreading
            total_circles = 0
            
            for i in range(num_segments):
                try:
                    start_sample = i * segment_length
                    end_sample = min(start_sample + segment_length, original_length)
                    
                    # Legge il numero di cerchi per questo segmento
                    num_circles = struct.unpack('H', data[offset:offset+2])[0]
                    offset += 2
                    total_circles += num_circles
                    
                    # Estrae i parametri dei cerchi
                    circles = []
                    for _ in range(num_circles):
                        circle, offset = CircleParameters.from_bytes(data, offset, precision)
                        circles.append(circle)
                    
                    # Ricostruisce il segmento audio dai cerchi
                    segment = self._reconstruct_from_circles(circles, end_sample - start_sample)
                    reconstructed_audio[start_sample:end_sample] = segment
                    
                    if debug and i % 100 == 0:
                        print(f"Decompresso segmento {i}/{num_segments} con {num_circles} cerchi")
                        
                except (struct.error, IndexError) as e:
                    print(f"Errore nella decompressione del segmento {i}: {e}")
                    # Continua con il prossimo segmento
                    
            if debug:
                print(f"Totale cerchi decodificati: {total_circles}")
        
        # Elimina valori anomali
        threshold = 10.0
        mask = np.abs(reconstructed_audio) > threshold
        if np.any(mask):
            if debug:
                print(f"Rilevati {np.sum(mask)} valori anomali. Sostituiti con 0.")
            reconstructed_audio[mask] = 0.0
        
        # Applica un filtro passa-basso per rimuovere eventuali artefatti
        try:
            from scipy import signal
            b, a = signal.butter(4, 0.3, 'low')
            reconstructed_audio = signal.filtfilt(b, a, reconstructed_audio)
        except Exception as e:
            if debug:
                print(f"Errore nell'applicazione del filtro: {e}")
        
        # Denormalizza l'audio
        reconstructed_audio *= normalization_factor
        
        # Clip per evitare overflow
        reconstructed_audio = np.clip(reconstructed_audio, -32767, 32767)
        
        # Converte in int16 per il salvataggio WAV
        audio_int16 = reconstructed_audio.astype(np.int16)
        
        # Salva il file WAV
        try:
            wavfile.write(output_file, sample_rate, audio_int16)
            print(f"Decompressione completata. File salvato: {output_file}")
        except Exception as e:
            print(f"Errore nel salvataggio del file WAV: {e}")
    
    def _approximate_with_circles(self, segment: np.ndarray) -> List[CircleParameters]:
        """
        Approssima un segmento audio con una serie di cerchi.
        
        Args:
            segment: segmento audio da approssimare
            
        Returns:
            Lista di CircleParameters che approssimano il segmento
        """
        circles = []
        x_position = 0.0
        remaining_segment = segment.copy()
        
        # Filtra il segmento per eliminare i valori molto bassi (rumore)
        silence_threshold = 0.005
        if np.max(np.abs(remaining_segment)) < silence_threshold:
            # Se il segmento è praticamente silenzio, non approssimare con cerchi
            return circles
        
        # Limita il numero di iterazioni per evitare loop infiniti
        max_iterations = 50
        iteration = 0
        
        # Dimensione minima del segmento da processare
        min_segment_size = 10
        
        while (len(remaining_segment) > min_segment_size and 
               np.max(np.abs(remaining_segment)) > self.min_accuracy and 
               iteration < max_iterations):
            
            iteration += 1
            
            # Trova il punto di massima ampiezza
            max_idx = np.argmax(np.abs(remaining_segment))
            y_value = remaining_segment[max_idx]
            
            # Salta se il valore è troppo piccolo
            if abs(y_value) < silence_threshold:
                break
                
            # Determina se usare la metà superiore o inferiore del cerchio
            use_upper_half = y_value >= 0
            
            # Calcola il raggio ottimale usando una ricerca più robusta
            window_size = min(200, len(remaining_segment) - max_idx)
            if window_size <= 5:  # Troppo piccolo per un'approssimazione valida
                break
                
            best_radius = 0
            min_error = float('inf')
            
            # Usa una spaziatura logaritmica per testare più valori piccoli
            for radius_candidate in np.logspace(-2, 0.5, 25):
                # Assicurati che il raggio sia sensato rispetto alla lunghezza della finestra
                if radius_candidate > window_size / 2:
                    continue
                    
                local_error = self._calculate_circle_error(
                    remaining_segment[max_idx:max_idx+window_size],
                    radius_candidate,
                    use_upper_half
                )
                
                if local_error < min_error:
                    min_error = local_error
                    best_radius = radius_candidate
            
            # Verifica che abbiamo trovato un raggio valido
            if best_radius <= 0.001 or best_radius > 10.0:
                # Raggio non plausibile, salta questa iterazione
                remaining_segment[max_idx] = 0  # Ignora questo punto nelle iterazioni future
                continue
            
            # Controlla se l'errore è accettabile
            if min_error > 0.5:  # Soglia arbitraria per errore accettabile
                # L'errore è troppo grande, probabilmente un'approssimazione scarsa
                remaining_segment[max_idx] = 0  # Ignora questo punto
                continue
            
            # Calcola e controlla l'offset
            x_offset = float(max_idx) - x_position if len(circles) > 0 else float(max_idx)
            if x_offset < 0:
                # Non dovrebbe mai accadere, ma per sicurezza
                x_offset = 0.1
            
            # Crea un cerchio solo se tutti i parametri sembrano validi
            circle = CircleParameters(best_radius, x_offset, use_upper_half)
            
            # Convalida che i parametri non siano duplicati di cerchi precedenti
            if len(circles) > 0:
                last_circle = circles[-1]
                if (abs(last_circle.radius - best_radius) < 0.01 and 
                    abs(x_offset) < 0.1 and 
                    last_circle.use_upper_half == use_upper_half):
                    # Questo cerchio è troppo simile all'ultimo, salta
                    remaining_segment[max_idx] = 0
                    continue
            
            circles.append(circle)
            
            # Aggiorna la posizione x
            x_position = float(max_idx)
            
            # Sottrai la forma del cerchio dal segmento rimanente
            circle_shape = self._generate_circle_shape(
                best_radius,
                window_size,
                use_upper_half
            )
            
            # Scala l'ampiezza della forma del cerchio
            if use_upper_half and y_value > 0:
                scale_factor = y_value
            elif not use_upper_half and y_value < 0:
                scale_factor = -y_value
            else:
                scale_factor = abs(y_value)

            circle.amplitude = scale_factor # Set the amplitude
            circle_shape *= scale_factor
            
            # Sottrai la forma e aggiorna il segmento rimanente
            if max_idx + window_size <= len(remaining_segment):
                remaining_segment[max_idx:max_idx+window_size] -= circle_shape
            else:
                remaining_segment[max_idx:] -= circle_shape[:len(remaining_segment)-max_idx]
            
            # Verifica la differenza massima rimanente
            if np.max(np.abs(remaining_segment)) < self.min_accuracy:
                break
        
        # Verifica finale per assicurarsi che abbiamo cerchi validi
        filtered_circles = []
        for circle in circles:
            # Filtra cerchi con parametri non validi
            if (0.001 < circle.radius < 10.0 and 
                0 <= circle.x_offset < len(segment)):
                filtered_circles.append(circle)
        
        return filtered_circles
    
    def _calculate_circle_error(self, segment: np.ndarray, radius: float, use_upper_half: bool) -> float:
        """Calcola l'errore tra un segmento e la forma di un cerchio."""
        circle_shape = self._generate_circle_shape(radius, len(segment), use_upper_half)
        
        # Scala la forma del cerchio per adattarla al primo punto del segmento
        scale_factor = segment[0] / circle_shape[0] if circle_shape[0] != 0 else 1.0
        circle_shape *= scale_factor
        
        # Calcola l'errore quadratico medio
        error = np.mean((segment - circle_shape) ** 2)
        return error
    
    def _generate_circle_shape(self, radius: float, length: int, use_upper_half: bool) -> np.ndarray:
        """Genera la forma di un cerchio lungo l'asse x."""
        x = np.linspace(0, radius*2, length)
        y = np.zeros(length)
        
        for i, xi in enumerate(x):
            if xi <= 2*radius:
                # Equazione del cerchio: (x - r)² + y² = r²
                discriminant = radius**2 - (xi - radius)**2
                if discriminant >= 0:
                    y_value = math.sqrt(discriminant)
                    y[i] = y_value if use_upper_half else -y_value
        
        return y
    
    def _reconstruct_from_circles(self, circles: List[CircleParameters], length: int) -> np.ndarray:
        """Ricostruisce un segmento audio dai parametri dei cerchi."""
        reconstructed = np.zeros(length, dtype=np.float32)
        current_x = 0.0
                # Validazione: se non ci sono cerchi validi, ritorna silenzio
        if not circles:
            return reconstructed
            
        for circle in circles:
            # Aggiorna la posizione corrente
            current_x += max(0, circle.x_offset)  # Assicura che l'offset sia positivo
            start_idx = int(current_x)
            
            # Salta se fuori dai limiti
            if start_idx >= length:
                continue
                
            # Validazione del raggio
            if circle.radius <= 0:
                continue
                
            # Determina la lunghezza della forma del cerchio
            circle_length = min(int(circle.radius * 2) + 1, length - start_idx)
            
            if circle_length <= 0:
                continue
            
            # Genera la forma del cerchio
            circle_shape = self._generate_circle_shape(
                circle.radius,
                circle_length,
                circle.use_upper_half
            )
            
            # Applica il fattore di ampiezza
            circle_shape *= circle.amplitude
            
            # Aggiungi la forma del cerchio al segmento ricostruito
            end_idx = min(start_idx + circle_length, length)
            shape_length = end_idx - start_idx
            
            # Verifica che gli indici siano validi
            if start_idx < 0:
                shape_offset = -start_idx
                start_idx = 0
                shape_length = min(shape_length - shape_offset, end_idx - start_idx)
                
                if shape_length <= 0:
                    continue
                    
                circle_shape = circle_shape[shape_offset:shape_offset+shape_length]
            
            try:
                reconstructed[start_idx:end_idx] += circle_shape[:shape_length]
            except ValueError as e:
                print(f"Errore nella ricostruzione: {e}")
                print(f"Dimensioni: start_idx={start_idx}, end_idx={end_idx}, shape_length={shape_length}, circle_shape.shape={circle_shape.shape}")
                # Continua con il prossimo cerchio
        
        # Applica clipping per evitare valori estremi
        np.clip(reconstructed, -1.0, 1.0, out=reconstructed)
        
        return reconstructed


def optimize_circles(circles: List[CircleParameters], max_error: float = 0.2) -> List[CircleParameters]:
    """
    Ottimizza una lista di cerchi rimuovendo quelli ridondanti o poco significativi.
    
    Args:
        circles: lista di cerchi da ottimizzare
        max_error: errore massimo consentito
        
    Returns:
        Lista ottimizzata di cerchi
    """
    if not circles:
        return []
        
    # Ordina i cerchi per ampiezza (dal più significativo al meno significativo)
    sorted_circles = sorted(circles, key=lambda c: abs(c.amplitude), reverse=True)
    
    # Inizia con il cerchio più significativo
    optimized = [sorted_circles[0]]
    
    # Per ogni cerchio, verifica se può essere unito a un cerchio esistente
    for i in range(1, len(sorted_circles)):
        circle = sorted_circles[i]
        
        # Se l'ampiezza è molto piccola, salta il cerchio
        if abs(circle.amplitude) < 0.02:
            continue
            
        # Cerca il cerchio più vicino in termini di posizione
        min_dist = float('inf')
        closest_idx = -1
        
        for j, existing in enumerate(optimized):
            # Calcola la distanza approssimativa (considerando anche offset)
            dist = abs(circle.x_offset - existing.x_offset)
            
            # Se sono vicini e hanno lo stesso segno, potrebbero essere combinati
            if (dist < min(circle.radius, existing.radius) * 2 and 
                circle.use_upper_half == existing.use_upper_half):
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j
        
        # Se c'è un cerchio vicino, prova a combinarlo
        if closest_idx >= 0 and min_dist < max_error:
            existing = optimized[closest_idx]
            
            # Combina i cerchi (media ponderata dei parametri)
            total_amp = abs(existing.amplitude) + abs(circle.amplitude)
            weight1 = abs(existing.amplitude) / total_amp
            weight2 = abs(circle.amplitude) / total_amp
            
            # Nuovo raggio (media ponderata)
            new_radius = existing.radius * weight1 + circle.radius * weight2
            
            # Nuovo offset (mantieni quello del cerchio principale)
            new_offset = existing.x_offset
            
            # Nuova ampiezza (somma)
            new_amplitude = existing.amplitude + circle.amplitude
            
            # Aggiorna il cerchio esistente
            optimized[closest_idx] = CircleParameters(
                new_radius, new_offset, existing.use_upper_half, new_amplitude
            )
        else:
            # Aggiungi il nuovo cerchio
            optimized.append(circle)
    
    # Ordina i cerchi per posizione
    optimized.sort(key=lambda c: c.x_offset)
    
    return optimized

def run_delta_encoding(compressor: CircularAudioCompressor, input_file: str, output_file: str):
    """
    Implementa una compressione con codifica delta per valori simili.
    
    Args:
        compressor: istanza del compressore
        input_file: percorso del file di input
        output_file: percorso del file di output
    """
    print("Avvio compressione con codifica delta...")
    
    # Legge il file audio
    sample_rate, audio_data = wavfile.read(input_file)
    
    # Assicura che i dati siano mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalizza l'audio tra -1 e 1
    audio_data = audio_data.astype(np.float32)
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        audio_data /= max_abs
    
    # Divide l'audio in segmenti
    segment_length = 500
    num_segments = math.ceil(len(audio_data) / segment_length)
    
    # Header standard
    header = struct.pack('I', sample_rate)
    header += struct.pack('I', len(audio_data))
    header += struct.pack('I', segment_length)
    header += struct.pack('I', num_segments)
    header += struct.pack('f', max_abs)
    header += struct.pack('B', 1 if compressor.precision == 'float16' else 2)
    header += struct.pack('B', 1)  # Flag per indicare che è usata la codifica delta
    
    # Comprime ogni segmento
    compressed_data = bytearray(header)
    
    # Prepara tabelle per la codifica delta
    radius_table = {}  # Mappa valori -> indici
    offset_table = {}
    amplitude_table = {}
    
    # Precisione per la chiave della tabella
    precision = 3  # Cifre decimali da mantenere
    
    for i in range(num_segments):
        start = i * segment_length
        end = min(start + segment_length, len(audio_data))
        segment = audio_data[start:end]
        
        # Approssima il segmento con cerchi
        circles = compressor._approximate_with_circles(segment)
        
        # Ottimizza i cerchi
        circles = optimize_circles(circles)
        
        # Serializza i cerchi usando la codifica delta
        segment_bytes = struct.pack('H', len(circles))
        
        for circle in circles:
            # Arrotonda i valori alla precisione specificata
            radius_key = round(circle.radius, precision)
            offset_key = round(circle.x_offset, precision)
            amplitude_key = round(circle.amplitude, precision)
            
            # Cerca i valori nelle tabelle o aggiungili
            if radius_key not in radius_table:
                radius_table[radius_key] = len(radius_table)
            radius_idx = radius_table[radius_key]
            
            if offset_key not in offset_table:
                offset_table[offset_key] = len(offset_table)
            offset_idx = offset_table[offset_key]
            
            if amplitude_key not in amplitude_table:
                amplitude_table[amplitude_key] = len(amplitude_table)
            amplitude_idx = amplitude_table[amplitude_key]
            
            # Flag per metà superiore/inferiore
            flag_byte = 1 if circle.use_upper_half else 0
            
            # Memorizza gli indici invece dei valori completi
            segment_bytes += struct.pack('H', radius_idx)
            segment_bytes += struct.pack('H', offset_idx)
            segment_bytes += struct.pack('H', amplitude_idx)
            segment_bytes += struct.pack('B', flag_byte)
        
        compressed_data.extend(segment_bytes)
    
    # Aggiungi le tabelle alla fine del file
    # Prima la dimensione delle tabelle
    compressed_data.extend(struct.pack('I', len(radius_table)))
    compressed_data.extend(struct.pack('I', len(offset_table)))
    compressed_data.extend(struct.pack('I', len(amplitude_table)))
    
    # Poi i valori delle tabelle
    pack_format = 'e' if compressor.precision == 'float16' else 'f'
    
    for radius_key, idx in sorted(radius_table.items(), key=lambda x: x[1]):
        compressed_data.extend(struct.pack(pack_format, float(radius_key)))
    
    for offset_key, idx in sorted(offset_table.items(), key=lambda x: x[1]):
        compressed_data.extend(struct.pack(pack_format, float(offset_key)))
    
    for amplitude_key, idx in sorted(amplitude_table.items(), key=lambda x: x[1]):
        compressed_data.extend(struct.pack(pack_format, float(amplitude_key)))
    
    # Applica compressione zlib
    try:
        import zlib
        compressed_data = zlib.compress(compressed_data, level=9)
        compressed_data = b'ZLIB' + compressed_data
    except ImportError:
        print("Modulo zlib non disponibile, nessuna compressione aggiuntiva applicata")
    
    # Salva il file compresso
    with open(output_file, 'wb') as f:
        f.write(compressed_data)
    
    # Stampa statistiche
    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    print(f"Compressione con codifica delta completata:")
    print(f"Dimensione originale: {original_size:,} bytes")
    print(f"Dimensione compressa: {compressed_size:,} bytes")
    print(f"Rapporto di compressione: {ratio:.2f}x")
    print(f"Dimensioni tabelle: {len(radius_table)}, {len(offset_table)}, {len(amplitude_table)}")
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': ratio,
        'tables_size': len(radius_table) + len(offset_table) + len(amplitude_table)
    }
def visualize_compression(input_file: str, output_dir: str = None, num_samples: int = 1000):
    """
    Visualizza il processo di compressione per un file audio.
    
    Args:
        input_file: percorso del file audio di input
        output_dir: cartella in cui salvare i grafici (se None, mostra i grafici)
        num_samples: numero di campioni da visualizzare
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.io import wavfile
        
        # Crea directory di output se necessario
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Legge il file audio
        sample_rate, audio_data = wavfile.read(input_file)
        
        # Assicura che i dati siano mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Normalizza l'audio
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 0:
            audio_data /= max_abs
            
        # Seleziona un segmento rappresentativo
        start_idx = len(audio_data) // 4  # Un quarto del file
        segment = audio_data[start_idx:start_idx+num_samples]
        
        # Crea un'istanza del compressore
        compressor = CircularAudioCompressor(precision='float32', min_accuracy=0.05)
        
        # Approssima il segmento con cerchi
        circles = compressor._approximate_with_circles(segment)
        
        # Ricostruisce il segmento
        reconstructed = compressor._reconstruct_from_circles(circles, len(segment))
        
        # Calcola l'errore
        error = segment - reconstructed
        max_error = np.max(np.abs(error))
        mse = np.mean(error**2)
        
        # Crea il grafico
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        
        # Originale
        axs[0].plot(segment, label='Originale')
        axs[0].set_title('Forma d\'onda originale')
        axs[0].set_ylabel('Ampiezza')
        axs[0].grid(True)
        axs[0].legend()
        
        # Ricostruito
        axs[1].plot(reconstructed, 'r-', label='Ricostruito')
        axs[1].set_title(f'Forma d\'onda ricostruita ({len(circles)} cerchi)')
        axs[1].set_ylabel('Ampiezza')
        axs[1].grid(True)
        axs[1].legend()
        
        # Errore
        axs[2].plot(error, 'g-', label='Errore')
        axs[2].set_title(f'Errore (MSE: {mse:.6f}, Max: {max_error:.6f})')
        axs[2].set_xlabel('Campioni')
        axs[2].set_ylabel('Ampiezza errore')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        
        # Salva o mostra il grafico
        if output_dir:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            plt.savefig(os.path.join(output_dir, f"{filename}_compression_analysis.png"), dpi=300)
            print(f"Grafico salvato in: {output_dir}/{filename}_compression_analysis.png")
        else:
            plt.show()
            
        # Visualizza anche i cerchi
        if len(circles) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Coordinate x cumulative per posizionare i cerchi
            x_positions = [0]
            for circle in circles[:-1]:
                x_positions.append(x_positions[-1] + circle.x_offset)
                
            # Disegna i cerchi
            for i, circle in enumerate(circles):
                x_center = x_positions[i] + circle.radius
                y_center = 0
                
                # Disegna il cerchio completo
                circle_patch = plt.Circle((x_center, y_center), circle.radius, 
                                         fill=False, alpha=0.7, 
                                         color='blue' if circle.use_upper_half else 'red')
                ax.add_patch(circle_patch)
                
                # Evidenzia la parte usata (superiore o inferiore)
                arc = plt.matplotlib.patches.Arc((x_center, y_center), 
                                              2*circle.radius, 2*circle.radius,
                                              theta1=0, theta2=180 if circle.use_upper_half else 360,
                                            #   start_angle=0, angle=180 if circle.use_upper_half else 0, # Removed unused parameters
                                              angle = 0,
                                              color='green', linewidth=2)
                ax.add_patch(arc)
            
            # Configura il grafico
            ax.set_xlim(0, max(x_positions[-1] + circles[-1].radius*2, num_samples))
            ax.set_ylim(-max(c.radius for c in circles)*1.2, max(c.radius for c in circles)*1.2)
            ax.set_title(f'Rappresentazione dei {len(circles)} cerchi usati per l\'approssimazione')
            ax.set_xlabel('Posizione (campioni)')
            ax.set_ylabel('Ampiezza')
            ax.grid(True)
            ax.set_aspect('equal')
            
            # Legenda
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label='Metà superiore'),
                Line2D([0], [0], color='red', lw=2, label='Metà inferiore'),
                Line2D([0], [0], color='green', lw=2, label='Parte utilizzata')
            ]
            ax.legend(handles=legend_elements)
            
            # Salva o mostra il grafico
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{filename}_circles.png"), dpi=300)
                print(f"Grafico dei cerchi salvato in: {output_dir}/{filename}_circles.png")
            else:
                plt.show()
        
        return {
            "num_circles": len(circles),
            "bytes_per_circle": 7 if compressor.precision == 'float16' else 13,  # 3 float + 1 byte
            "original_bytes": len(segment) * 2,  # 16-bit audio
            "compressed_bytes": len(circles) * (7 if compressor.precision == 'float16' else 13),
            "mse": mse,
            "max_error": max_error,
            "compression_ratio": (len(segment) * 2) / (len(circles) * (7 if compressor.precision == 'float16' else 13))
        }
        
    except ImportError:
        print("Impossibile visualizzare la compressione: matplotlib o scipy non installati")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compressione e decompressione audio circolare')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['compress', 'decompress', 'analyze', 'visualize', 'delta'],
                        help='Modalità: compress, decompress, analyze, visualize o delta')
    parser.add_argument('--input', type=str, required=True,
                        help='File di input')
    parser.add_argument('--output', type=str,
                        help='File di output (non necessario per analyze/visualize)')
    parser.add_argument('--precision', type=str, default='float16', 
                        choices=['float16', 'float32'],
                        help='Precisione dei float (default: float16)')
    parser.add_argument('--accuracy', type=float, default=0.05,
                        help='Precisione minima dell\'approssimazione (default: 0.05)')
    parser.add_argument('--no-threading', action='store_true',
                        help='Disabilita il multithreading')
    parser.add_argument('--debug', action='store_true',
                        help='Abilita output di debug')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Numero di campioni da visualizzare (per visualize)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory per salvare le visualizzazioni (per visualize)')
    
    args = parser.parse_args()
    
    compressor = CircularAudioCompressor(
        precision=args.precision, 
        min_accuracy=args.accuracy,
        use_threading=not args.no_threading
    )
    
    if args.mode == 'compress':
        if not args.output:
            print("Errore: parametro --output richiesto per la compressione")
            exit(1)
        compressor.compress(args.input, args.output, debug=args.debug)
    
    elif args.mode == 'decompress':
        if not args.output:
            print("Errore: parametro --output richiesto per la decompressione")
            exit(1)
        compressor.decompress(args.input, args.output, debug=args.debug)
    
    elif args.mode == 'delta':
        if not args.output:
            print("Errore: parametro --output richiesto per la compressione delta")
            exit(1)
        run_delta_encoding(compressor, args.input, args.output)
        
    elif args.mode == 'visualize':
        # Visualizza il processo di compressione
        stats = visualize_compression(args.input, args.output_dir, args.samples)
        if stats:
            print("\nStatistiche di compressione:")
            print(f"  Numero di cerchi: {stats['num_circles']}")
            print(f"  Bytes per cerchio: {stats['bytes_per_circle']}")
            print(f"  Bytes originali: {stats['original_bytes']}")
            print(f"  Bytes compressi: {stats['compressed_bytes']}")
            print(f"  Rapporto di compressione: {stats['compression_ratio']:.2f}x")
            print(f"  Errore quadratico medio: {stats['mse']:.6f}")
            print(f"  Errore massimo: {stats['max_error']:.6f}")
        
    elif args.mode == 'analyze':
        # Analizza il file compresso
        print(f"Analisi del file compresso: {args.input}")
        
        with open(args.input, 'rb') as f:
            data = f.read()
            
        # Controlla se ci sono pattern ripetuti nel file
        from collections import Counter
        
        # Divide il file in blocchi di 4 byte e conta le occorrenze
        chunks = [data[i:i+4] for i in range(0, len(data), 4)]
        chunk_counts = Counter(chunks)
        
        # Mostra i 10 pattern più comuni
        print("\nPattern più comuni (blocchi di 4 byte):")
        for chunk, count in chunk_counts.most_common(10):
            hex_repr = ' '.join(f"{b:02X}" for b in chunk)
            percentage = (count / len(chunks)) * 100
            print(f"  {hex_repr}: {count} occorrenze ({percentage:.2f}%)")
            
        # Stima dell'entropia (misura dell'informazione)
        from math import log2
        
        entropy_chunks = 0
        total_chunks = sum(chunk_counts.values())
        for count in chunk_counts.values():
            p = count / total_chunks
            entropy_chunks -= p * log2(p)
            
        max_entropy = log2(min(len(chunk_counts), 2**(4*8)))
        entropy_ratio = entropy_chunks / max_entropy if max_entropy > 0 else 0
        
        print(f"\nEntropia del file: {entropy_chunks:.2f} bit/chunk")
        print(f"Entropia massima possibile: {max_entropy:.2f} bit/chunk")
        print(f"Rapporto entropia: {entropy_ratio:.2%}")
        
        if entropy_ratio < 0.7:
            print("\nIl file ha un'entropia relativamente bassa, suggerendo che potrebbe essere ulteriormente compresso.")
        else:
            print("\nIl file ha un'entropia elevata, suggerendo che la compressione è già efficiente.")
        
        # Se il file ha l'header ZLIB, mostra anche l'entropia del file decompresso
        decompressed = None  # Initialize decompressed
        if data[:4] == b'ZLIB':
            try:
                import zlib
                decompressed = zlib.decompress(data[4:])
                
                # Analizza i dati decompressi
                chunks_decomp = [decompressed[i:i+4] for i in range(0, len(decompressed), 4)]
                chunk_counts_decomp = Counter(chunks_decomp)
                
                entropy_decomp = 0
                total_chunks_decomp = sum(chunk_counts_decomp.values())
                for count in chunk_counts_decomp.values():
                    p = count / total_chunks_decomp
                    entropy_decomp -= p * log2(p)
                
                print(f"\nDati zlib decompressi:")
                print(f"  Dimensione: {len(decompressed):,} bytes")
                print(f"  Entropia: {entropy_decomp:.2f} bit/chunk")
                
                # Esempio di valori ripetuti nei dati decompressi
                print("\nPattern più comuni nei dati decompressi:")
                for chunk, count in chunk_counts_decomp.most_common(5):
                    hex_repr = ' '.join(f"{b:02X}" for b in chunk)
                    percentage = (count / len(chunks_decomp)) * 100
                    print(f"  {hex_repr}: {count} occorrenze ({percentage:.2f}%)")
                    
            except (ImportError, zlib.error) as e:
                print(f"Errore nella decompressione zlib: {e}")
                
        # Prova a estrarre l'header se possibile
        try:
            # Se il file è compresso con zlib, decomprimilo prima
            header_data = decompressed if data[:4] == b'ZLIB' else data
            
            offset = 0
            sample_rate = struct.unpack('I', header_data[offset:offset+4])[0]
            offset += 4
            
            original_length = struct.unpack('I', header_data[offset:offset+4])[0]
            offset += 4
            
            segment_length = struct.unpack('I', header_data[offset:offset+4])[0]
            offset += 4
            
            num_segments = struct.unpack('I', header_data[offset:offset+4])[0]
            offset += 4
            
            normalization_factor = struct.unpack('f', header_data[offset:offset+4])[0]
            offset += 4
            
            precision_code = struct.unpack('B', header_data[offset:offset+1])[0]
            precision = 'float16' if precision_code == 1 else 'float32'
            offset += 1
            
            is_delta_encoded = False  # Initialize is_delta_encoded
            if len(header_data) > offset:  # Check if there are more bytes (for delta flag)
                delta_flag = struct.unpack('B', header_data[offset:offset+1])[0]
                is_delta_encoded = (delta_flag == 1)
                offset +=1

            print("\nInformazioni estratte dall'header:")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Lunghezza originale: {original_length:,} campioni ({original_length/sample_rate:.2f} secondi)")
            print(f"  Lunghezza segmento: {segment_length} campioni")
            print(f"  Numero segmenti: {num_segments}")
            print(f"  Fattore normalizzazione: {normalization_factor}")
            print(f"  Precisione: {precision}")
            if is_delta_encoded:
                print("  Codifica Delta: Attiva")
            
            # Calcola la dimensione attesa per i dati
            expected_size = len(header_data) - offset
            print(f"  Dimensione dati (escluso header): {expected_size:,} bytes")
            if num_segments > 0: #Avoid ZeroDivisionError
               print(f"  Dimensione media per segmento: {expected_size/num_segments:.1f} bytes")
            
            # Analisi dei cerchi
            try:
                total_circles = 0
                circle_sizes = []
                
                # Legge i primi N segmenti per analisi
                max_segments_to_analyze = min(100, num_segments)
                print(f"\nAnalisi dei cerchi nei primi {max_segments_to_analyze} segmenti:")
                
                for i in range(max_segments_to_analyze):
                    try:
                        # Legge il numero di cerchi per questo segmento
                        num_circles = struct.unpack('H', header_data[offset:offset+2])[0]
                        offset += 2
                        total_circles += num_circles
                        circle_sizes.append(num_circles)
                        
                        # Salta i dati dei cerchi
                        circle_data_size = 0
                        if precision == 'float16':
                            circle_data_size = num_circles * (2*3 + 1)  # 3 float16 e 1 byte per cerchio
                        else:
                            circle_data_size = num_circles * (4*3 + 1)  # 3 float32 e 1 byte per cerchio
                            
                        offset += circle_data_size
                        
                    except (struct.error, IndexError) as e:
                        print(f"Errore nell'analisi del segmento {i}: {e}")
                        break
                
                if circle_sizes:
                    avg_circles = sum(circle_sizes) / len(circle_sizes)
                    max_circles = max(circle_sizes)
                    min_circles = min(circle_sizes)
                    
                    print(f"  Media cerchi per segmento: {avg_circles:.2f}")
                    print(f"  Massimo cerchi in un segmento: {max_circles}")
                    print(f"  Minimo cerchi in un segmento: {min_circles}")
                    print(f"  Totale cerchi analizzati: {total_circles}")
                    
                    # Suggerimenti per ottimizzazione
                    if avg_circles < 1:
                        print("\nSuggerimento: La maggior parte dei segmenti non contiene cerchi.")
                        print("  Prova ad aumentare la lunghezza del segmento o ridurre la precisione minima.")
                    elif avg_circles > 20:
                        print("\nSuggerimento: I segmenti contengono molti cerchi.")
                        print("  Prova ad aumentare la precisione minima per ridurre il numero di cerchi.")

                     # Stima l'efficienza della compressione
                    bytes_per_circle = 7 if precision == 'float16' else 13
                    estimated_total_circles = avg_circles * num_segments
                    estimated_compressed_size = estimated_total_circles * bytes_per_circle + offset
                    estimated_original_size = original_length * 2  # 16-bit audio
                    if estimated_compressed_size > 0:
                        estimated_ratio = estimated_original_size / estimated_compressed_size
                    else:
                        estimated_ratio = float('inf') # Avoid division by zero

                    print(f"\nStima compressione totale:")
                    print(f"  Cerchi totali stimati: {estimated_total_circles:.0f}")
                    print(f"  Dimensione compressa stimata: {estimated_compressed_size:,.0f} bytes")
                    print(f"  Dimensione originale: {estimated_original_size:,.0f} bytes")
                    print(f"  Rapporto di compressione stimato: {estimated_ratio:.2f}x")

                    if estimated_ratio < 1:
                        print("\nATTENZIONE: Il file compresso potrebbe essere più grande dell'originale.")
                        print("  Considera di aumentare la precisione minima o usare un altro metodo di compressione.")

            except Exception as e:
                print(f"Errore nell'analisi dei cerchi: {e}")
            
        except (struct.error, IndexError, NameError) as e:
            print(f"\nImpossibile estrarre l'header: {e}")

        # Se è richiesto il debug, mostra l'analisi esadecimale più dettagliata
        if args.debug:
            # Mostra gruppi ripetuti di byte più lunghi
            print("\nRicerca di pattern ripetuti più lunghi...")
            
            for pattern_len in [8, 16, 32]:
                if len(data) < pattern_len * 2:
                    continue
                    
                pattern_chunks = [data[i:i+pattern_len] for i in range(0, len(data) - pattern_len, pattern_len)]
                pattern_counts = Counter(pattern_chunks).most_common(3)
                
                if pattern_counts and pattern_counts[0][1] > 1:
                    print(f"\nPattern ripetuti di {pattern_len} bytes:")
                    for pattern, count in pattern_counts:
                        if count > 1:  # Mostra solo se ripetuto
                            hex_repr = ' '.join(f"{b:02X}" for b in pattern)
                            percentage = (count * pattern_len / len(data)) * 100
                            print(f"  Ripetizioni: {count}, {percentage:.2f}% del file")
                            print(f"  Pattern: {hex_repr}")
            
            # Analisi della distribuzione dei valori per byte
            print("\nDistribuzione dei valori per posizione byte:")
            
            # Analizza i primi N byte del file per vedere se hanno distribuzioni diverse
            bytes_to_analyze = min(32, len(data))
            for i in range(bytes_to_analyze):
                if i % 4 == 0:  # Analizza solo il primo byte di ogni word per brevità
                    vals = [data[j] for j in range(i, len(data), bytes_to_analyze) if j < len(data)]
                    val_counts = Counter(vals)
                    unique_vals = len(val_counts)
                    
                    print(f"  Byte {i}: {unique_vals} valori unici")
                    if unique_vals <= 5:  # Se ci sono pochi valori, mostrali
                        for val, count in val_counts.most_common(5):
                            percentage = (count / len(vals)) * 100
                            print(f"    0x{val:02X}: {count} occorrenze ({percentage:.1f}%)")
            
            # Se è un file compresso, controlla per dati potenzialmente duplicati
            if decompressed and len(decompressed) > 1000:
                print("\nVerifica di potenziali duplicazioni nei dati dei cerchi:")
                
                # Cerca segmenti identici nella parte dei dati (dopo l'header)
                data_part = decompressed[offset:]
                segment_size = 7 if precision == 'float16' else 13  # dimensione approssimativa di un cerchio
                
                # Campiona alcuni segmenti per confrontarli
                sample_size = 100
                samples = []
                for i in range(0, min(1000, len(data_part) - segment_size), segment_size):
                    samples.append(data_part[i:i+segment_size])
                
                # Conta le occorrenze
                sample_counts = Counter(samples)
                duplicates = [(sample, count) for sample, count in sample_counts.items() if count > 1]
                
                if duplicates:
                    print(f"  Trovati {len(duplicates)} segmenti duplicati nei primi 1000 campioni")
                    for sample, count in sorted(duplicates, key=lambda x: x[1], reverse=True)[:3]:
                        hex_repr = ' '.join(f"{b:02X}" for b in sample)
                        print(f"    '{hex_repr}': {count} occorrenze")
                    
                    print("\nSuggerimento: Considera l'implementazione di una tabella di lookup per valori ripetuti")
                else:
                    print("  Nessuna duplicazione significativa rilevata nei primi 1000 campioni")