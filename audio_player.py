import sounddevice as sd
import soundfile as sf
import threading
from scipy.signal import resample, butter, lfilter
import numpy as np
import librosa
import os


def calculate_tempo(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo) if isinstance(tempo, np.ndarray) else float(tempo), _
    except Exception as e:
        print(f"Error calculating tempo for {file_path}: {e}")
        return "N/A", "N/A"


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a


def apply_filter(data, fs, band, gain):
    if band == 'low':
        b, a = butter_lowpass(200, fs)
    elif band == 'mid':
        b, a = butter_bandpass(200, 2000, fs)
    elif band == 'high':
        b, a = butter_highpass(2000, fs)
    else:
        return data
    filtered = lfilter(b, a, data)
    return gain * filtered


class AudioPlayer:
    def __init__(self, name):
        self.beats = None
        self.beat_times = None
        self.name = name
        self.file_path = None
        self.json_path = None
        self.audio_data = None
        self.samplerate = None
        self.playing = False
        self.tempo = 1
        self.stop_event = threading.Event()
        self.current_index = 0
        self.tempo_factor = 1.0
        self.gains = {'low': 1.0, 'mid': 1.0, 'high': 1.0}
        self.audio_data_lr = None
        self.sr_lr = None
        self.duration = None

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        if lowcut >= highcut:
            raise ValueError("lowcut must be less than highcut")
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return b, a

    def butter_highpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high')
        return b, a

    def apply_filter(self, data, fs, band, gain):
        if gain == 1.0:
            return data
        if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype, np.floating):
            raise ValueError(f"Dane wejściowe muszą być tablicą liczb zmiennoprzecinkowych, otrzymano: {data.dtype}")
        if band == 'low':
            b, a = self.butter_lowpass(200, fs)
        elif band == 'mid':
            b, a = self.butter_bandpass(200, 2000, fs)
        elif band == 'high':
            b, a = self.butter_highpass(2000, fs)
        else:
            return data
        filtered = lfilter(b, a, data)
        return (gain * filtered).astype('float32')  # Konwersja do float32

    def load_track(self, file_path):
        try:
            self.tempo_factor = 1
            self.audio_data, self.samplerate = sf.read(file_path, dtype='float32')
            self.file_path = file_path
            self.json_path = f'waveforms/{os.path.splitext(os.path.basename(file_path))[0]}.json'
            self.current_index = 0
            self.tempo, self.beats = calculate_tempo(self.file_path)
            self.audio_data_lr, self.sr_lr = librosa.load(self.file_path, sr=None)
            self.duration = librosa.get_duration(y=self.audio_data_lr, sr=self.sr_lr)
            self.beat_times = librosa.frames_to_time(self.beats, sr=self.sr_lr)
            print(f"Załadowano plik: {file_path}")
        except Exception as e:
            print(f"Błąd ładowania pliku: {e}")

    def change_tempo_factor(self, factor):
        self.tempo_factor = float(factor)
        # print(f"Ustawiono tempo na  {self.name}: {factor*self.tempo}")

    def change_tempo(self, tempo):
        self.tempo_factor = float(tempo) / self.tempo
        # print(f"Ustawiono tempo na  {self.name}: {factor*self.tempo}")

    def sync_to(self, other_player):
        if other_player.tempo != "N/A":
            self.change_tempo(other_player.tempo * other_player.tempo_factor)

        print(other_player.tempo * other_player.tempo_factor)

    def _play_audio(self):
        try:
            original_audio = self.audio_data.copy()
            self.stop_event.clear()
            self.playing = True
            blocksize = 1024
            channels = original_audio.shape[1] if original_audio.ndim > 1 else 1

            print(f"Rozpoczynanie odtwarzania {self.name}")
            with sd.OutputStream(samplerate=self.samplerate, channels=channels) as stream:
                while self.current_index < len(original_audio):
                    if self.stop_event.is_set():
                        break

                    end_index = self.current_index + blocksize
                    block = original_audio[self.current_index:end_index]

                    if channels > 1:  # Stereo
                        filtered_blocks = []
                        for ch in range(channels):
                            filtered = (
                                    self.apply_filter(block[:, ch], self.samplerate, 'low', self.gains['low']) +
                                    self.apply_filter(block[:, ch], self.samplerate, 'mid', self.gains['mid']) +
                                    self.apply_filter(block[:, ch], self.samplerate, 'high', self.gains['high'])
                            )
                            resampled = resample(filtered, int(len(filtered) / self.tempo_factor))
                            filtered_blocks.append(resampled)

                        min_length = min(map(len, filtered_blocks))
                        block = np.column_stack([b[:min_length] for b in filtered_blocks])

                    else:  # Mono
                        block = (
                                self.apply_filter(block, self.samplerate, 'low', self.gains['low']) +
                                self.apply_filter(block, self.samplerate, 'mid', self.gains['mid']) +
                                self.apply_filter(block, self.samplerate, 'high', self.gains['high'])
                        )
                        block = resample(block, int(len(block) / self.tempo_factor))

                    block = block.astype('float32')
                    stream.write(block)
                    self.current_index += blocksize
        except Exception as e:
            print(f"Błąd odtwarzania: {e}")
        finally:
            self.playing = False
            print("Zakończono odtwarzanie.")

    def start(self):
        if self.audio_data is None:
            print("Nie załadowano pliku.")
            return

        if not self.playing:
            self.thread = threading.Thread(target=self._play_audio, daemon=True)
            self.thread.start()
        else:
            print("Plik już jest odtwarzany.")

    def stop(self):
        if self.playing:
            self.stop_event.set()
            self.current_index = 0
            print("Odtwarzanie zatrzymane i zresetowane.")
        else:
            print("Brak aktywnego odtwarzania do zatrzymania.")

    def pause(self):
        if self.playing:
            self.stop_event.set()
            print("Odtwarzanie wstrzymane.")
        else:
            print("Brak aktywnego odtwarzania do pauzowania.")

    def set_gain(self, band, gain):
        if band in self.gains:
            self.gains[band] = np.float32(gain)
            print(f"Ustawiono wzmocnienie {band} na {gain}")
        else:
            print("Nieprawidłowe pasmo.")

    def dynamic_mix(self, other_player, fade_duration=30, blocksize=1024):
        try:
            other_player.sync_to(self)

            current_time = self.current_index / self.samplerate
            player_beats = np.array(self.beat_times)
            other_player_beats = np.array(other_player.beat_times)

            closest_beat_self = player_beats[np.argmin(np.abs(player_beats - current_time))]
            closest_beat_other = other_player_beats[np.argmin(np.abs(other_player_beats - closest_beat_self))]

            offset = closest_beat_self - closest_beat_other
            other_player.current_index += int(offset * other_player.samplerate)

            fade_samples = int(fade_duration * self.samplerate)
            fade_out = np.linspace(1, 0, fade_samples)[:, None]
            fade_in = np.linspace(0, 1, fade_samples)[:, None]

            def crossfade():
                with sd.OutputStream(samplerate=self.samplerate, channels=self.audio_data.shape[1]) as self_stream, \
                        sd.OutputStream(samplerate=other_player.samplerate,
                                        channels=other_player.audio_data.shape[1]) as other_stream:

                    while self.current_index < len(self.audio_data) and other_player.current_index < len(
                            other_player.audio_data):
                        self_block = self.audio_data[self.current_index:self.current_index + blocksize]
                        other_block = other_player.audio_data[
                                      other_player.current_index:other_player.current_index + blocksize]

                        min_length = min(len(self_block), len(other_block))
                        self_block = self_block[:min_length]
                        other_block = other_block[:min_length]

                        fade_index = self.current_index - int(closest_beat_self * self.samplerate)
                        if 0 <= fade_index < fade_samples:
                            fade_out_block = fade_out[fade_index:fade_index + len(self_block)]
                            fade_in_block = fade_in[fade_index:fade_index + len(other_block)]

                            self_block *= fade_out_block
                            other_block *= fade_in_block

                        self_block = self_block.astype(np.float32)
                        other_block = other_block.astype(np.float32)
                        self_stream.write(self_block)
                        other_stream.write(other_block)

                        self.current_index += blocksize
                        other_player.current_index += blocksize

            threading.Thread(target=crossfade, daemon=True).start()

            print("Dynamiczny crossfade uruchomiony!")

        except Exception as e:
            print(f"Błąd podczas dynamicznego miksowania: {e}")
