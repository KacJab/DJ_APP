from pydub import AudioSegment
import os
from pathlib import Path
import librosa
import numpy as np
import json


def convert_mp3_to_wav(input_path: str, output_dir: str = 'utils/'):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Plik nie istnieje: {input_path}")

    if not input_path.lower().endswith(".mp3"):
        raise ValueError("Plik wejściowy musi być w formacie MP3")

    output_path = output_dir + Path(input_path).stem + ".wav"

    try:
        audio = AudioSegment.from_mp3(input_path)
        audio.export(output_path, format="wav")
        print(f"Plik zostal skonwertowany: {output_path}")
        return output_path

    except Exception as e:
        raise RuntimeError(f"Wystąpił blad podczas konwersji: {e}")


def prepare_music_files(source_dir='utils', target_dir='waveforms'):
    missing_files = []

    for file in os.listdir(source_dir):
        if file.endswith(".wav"):
            json_filename = os.path.splitext(file)[0] + ".json"
            json_filepath = os.path.join(target_dir, json_filename)
            if not os.path.exists(json_filepath):
                missing_files.append((file, json_filepath))

    if missing_files:
        print("Brakujące pliki .json dla:")
        for file, json_filepath in missing_files:
            precompute_waveform_and_save_it_to_file(os.path.join(source_dir, file), json_filepath)
            print(f"Stworzono waveform dla {file}")
    else:
        print("Wszystkie pliki .wav mają odpowiadające pliki .json.")


def precompute_waveform_and_save_it_to_file(file, json_filepath):
    audio_data, sr = librosa.load(file, sr=None)
    duration = librosa.get_duration(y=audio_data, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    num_points = 10000  # Number of points for the entire waveform
    downsampled_audio = librosa.resample(audio_data, orig_sr=sr, target_sr=num_points / duration)
    times = np.linspace(0, duration, num=len(downsampled_audio))
    amplitude = downsampled_audio / np.max(np.abs(audio_data))  # Normalize amplitude
    waveform_data = {
        "times": times.tolist(),
        "amplitude": amplitude.tolist(),
        "beat_times": beat_times.tolist()
    }
    with open(json_filepath, "w") as f:
        json.dump(waveform_data, f)
    print(f"Precomputed waveform data saved to {json_filepath}.")


def load_json_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
