import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas
from track_manager import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display
import threading
import soundfile as sf

from audio_player import AudioPlayer as ap
from track_manager import load_json_data


class AudioMixerGUI:
    def __init__(self, root, player1, player2):
        self.root = root
        self.root.title("Audio Mixer")
        self.player1 = player1
        self.player2 = player2

        self.window_duration = 10
        self.min_window = 5
        self.max_window = 40
        self.current_start = 0

        self.canvas_width = 800
        self.canvas_height = 100
        self.canvas1 = Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas1.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.canvas2 = Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas2.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.frame_player1 = ttk.LabelFrame(root, text="Odtwarzacz 1")
        self.frame_player1.grid(row=2, column=0, padx=10, pady=10)

        self.frame_player2 = ttk.LabelFrame(root, text="Odtwarzacz 2")
        self.frame_player2.grid(row=2, column=1, padx=10, pady=10)

        self.create_player_controls(self.frame_player1, player1, player2)

        self.create_player_controls(self.frame_player2, player2, player1)

    def create_player_controls(self, frame, player, other_player):
        player.track_label = tk.Label(frame, text=Path(player.file_path).stem, fg="red", wraplength=200)
        player.track_label.grid(row=2, column=0, columnspan=3, pady=5)

        if player.name == 'player1':
            self.toggle_animation = self.animate(self.canvas1, load_json_data(player.json_path), self.window_duration,
                                                 self.canvas_width, self.canvas_height, 0.05, 50)

            btn_start = ttk.Button(frame, text="Start", command=self.toggle_start(self.toggle_animation, player))
            btn_start.grid(row=3, column=0, padx=5, pady=5)

            btn_stop = ttk.Button(frame, text="Stop", command=self.toggle_stop(self.toggle_animation, player))
            btn_stop.grid(row=3, column=1, padx=5, pady=5)
        else:
            self.toggle_animation2 = self.animate(self.canvas2, load_json_data(player.json_path), self.window_duration,
                                                  self.canvas_width, self.canvas_height, 0.05, 50)

            btn_start = ttk.Button(frame, text="Start", command=self.toggle_start(self.toggle_animation2, player))
            btn_start.grid(row=3, column=0, padx=5, pady=5)

            btn_stop = ttk.Button(frame, text="Stop", command=self.toggle_stop(self.toggle_animation2, player))
            btn_stop.grid(row=3, column=1, padx=5, pady=5)

        btn_sync = ttk.Button(frame, text="Sync Tempo", command=lambda: self.sync_tempo(player, other_player))
        btn_sync.grid(row=3, column=2, padx=5, pady=5)

        tk.Label(frame, text="Bass").grid(row=4, column=0)
        tk.Scale(frame, from_=-3, to=3, orient="vertical", resolution=0.1,
                 command=lambda value: player.set_gain("low", value)).grid(row=5, column=0)

        tk.Label(frame, text="Mid").grid(row=4, column=1)
        tk.Scale(frame, from_=-3, to=3, orient="vertical", resolution=0.1,
                 command=lambda value: player.set_gain("mid", value)).grid(row=5, column=1)

        tk.Label(frame, text="Treble").grid(row=4, column=2)
        tk.Scale(frame, from_=-3, to=3, orient="vertical", resolution=0.1,
                 command=lambda value: player.set_gain("high", value)).grid(row=5, column=2)

        tk.Label(frame, text="Tempo").grid(row=6, column=0, columnspan=3)
        player.tempo_scale = tk.Scale(frame, from_=0.5, to=2, orient="horizontal", length=200,
                                      resolution=0.01,
                                      label=f"Adjusted BPM: {round(player.tempo * player.tempo_factor, 2)}" if player.tempo != "N/A" else "BPM: N/A",
                                      command=lambda value: self.tempo_scale_method(value, player))
        player.tempo_scale.set(1)
        player.tempo_scale.grid(row=7, column=0, columnspan=3)
        print(player.tempo * player.tempo_factor)
        setattr(self, f"{player.name}_tempo_scale", player.tempo_scale)

        btn_mix = ttk.Button(frame, text="Mix Track", command=lambda: player.dynamic_mix(other_player))
        btn_mix.grid(row=8, column=0, columnspan=3, pady=10)

        btn_load = ttk.Button(frame, text="Load Track", command=lambda: self.load_track(player))
        btn_load.grid(row=9, column=0, columnspan=3, pady=5)

        tk.Label(frame, text="Pliki w folderze 'utils':").grid(row=10, column=0, columnspan=3)
        listbox = tk.Listbox(frame, height=6, width=40)
        listbox.grid(row=11, column=0, columnspan=3, pady=5)
        listbox.bind("<<ListboxSelect>>", lambda event: self.on_select(event, player))

        self.fill_file_list(listbox)

    def draw_waveform(self, canvas, data, start_time, window_duration, width, height):
        canvas.delete("all")

        times = np.array(data["times"])
        amplitude = np.array(data["amplitude"])
        beat_times = data["beat_times"]

        amplitude = amplitude / np.max(np.abs(amplitude))

        end_time = start_time + window_duration
        mask = (times >= start_time) & (times <= end_time)
        visible_times = times[mask]
        visible_amplitudes = amplitude[mask]

        if len(visible_times) == 0:
            return

        red_line_x = width / 2  # Czerwona linia
        x_coords = np.interp(visible_times, (start_time, end_time), (red_line_x, width + red_line_x))
        y_coords = np.interp(visible_amplitudes, (-1, 1), (height / 2, 0))
        y_coords_neg = np.interp(visible_amplitudes, (-1, 1), (height / 2, height))

        for i in range(1, len(x_coords)):
            canvas.create_line(x_coords[i - 1], y_coords[i - 1], x_coords[i], y_coords[i], fill="cyan", width=2)
            canvas.create_line(x_coords[i - 1], y_coords_neg[i - 1], x_coords[i], y_coords_neg[i], fill="cyan", width=2)

        left_coords = x_coords - red_line_x
        for i in range(1, len(left_coords)):
            if left_coords[i - 1] >= 0 and left_coords[i] >= 0:
                canvas.create_line(left_coords[i - 1], y_coords[i - 1], left_coords[i], y_coords[i], fill="cyan",
                                   width=2)
                canvas.create_line(left_coords[i - 1], y_coords_neg[i - 1], left_coords[i], y_coords_neg[i],
                                   fill="cyan", width=2)

        canvas.create_line(red_line_x, 0, red_line_x, height, fill="red", width=2)

        for beat in beat_times:
            if start_time <= beat <= end_time:
                x = np.interp(beat, (start_time, end_time), (red_line_x, width + red_line_x))
                canvas.create_line(x, 0, x, height, fill="green", width=1, dash=(4, 4))

        for beat in beat_times:
            if start_time - window_duration <= beat < start_time:
                x = np.interp(beat, (start_time - window_duration, start_time), (-red_line_x, red_line_x))
                if 0 <= x <= red_line_x:
                    canvas.create_line(x, 0, x, height, fill="green", width=1, dash=(4, 4))

    def animate(self, canvas, data, window_duration, width, height, step, interval):
        start_time = 0
        running = [False]

        self.draw_waveform(canvas, data, start_time, window_duration, width, height)

        def update():
            nonlocal start_time
            if running[0]:
                self.draw_waveform(canvas, data, start_time, window_duration, width, height)
                start_time += step
                canvas.after(interval, update)

        def toggle_running():
            running[0] = not running[0]
            if running[0]:
                print("Animacja uruchomiona")
                update()
            else:
                print("Animacja zatrzymana")

        return toggle_running

    def toggle_start(self, toggle_animation, player):
        def toggle():
            self.start_audio(player)
            toggle_animation()

        return toggle

    def toggle_stop(self, toggle_animation, player):
        def toggle():
            self.stop_audio(player)
            toggle_animation()

        return toggle

    def tempo_scale_method(self, value, player):
        player.change_tempo_factor(value)
        player.tempo_scale.config(
            label=f"Adjusted BPM: {round(player.tempo * player.tempo_factor, 2)}" if player.tempo != "N/A" else "BPM: N/A")

    def on_select(self, event, player):
        listbox = event.widget
        selected_index = listbox.curselection()
        if selected_index:
            if player.name == 'player1':
                self.toggle_animation = self.animate(self.canvas1, load_json_data(player.json_path),
                                                     self.window_duration,
                                                     self.canvas_width, self.canvas_height, 0.05, 50)
            else:
                self.toggle_animation2 = self.animate(self.canvas2, load_json_data(player.json_path),
                                                      self.window_duration,
                                                      self.canvas_width, self.canvas_height, 0.05, 50)
            selected_item = listbox.get(selected_index[0])
            print(f"Wybrano: {selected_item}")
            player.load_track('utils/' + selected_item)
            player.track_label.config(text=Path(player.file_path).stem)
            player.tempo_scale.config(
                label=f"Adjusted BPM: {round(player.tempo * player.tempo_factor, 2)}" if player.tempo != "N/A" else "BPM: N/A")
            player.tempo_scale.set(1)
            print("player1 zmieniony")

    def fill_file_list(self, listbox):
        folder_path = "./utils"
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                listbox.insert(tk.END, file)
        else:
            listbox.insert(tk.END, "Brak folderu 'utils'")

    def start_audio(self, player):
        player.start()
        print(f"Start odtwarzania: {player.name}")

    def stop_audio(self, player):
        player.pause()
        print(f"Stop odtwarzania: {player.name}")

    def sync_tempo(self, player, other_player):
        player.sync_to(other_player)
        player.tempo_scale.config(
            label=f"Adjusted BPM: {round(player.tempo * player.tempo_factor, 2)}" if player.tempo != "N/A" else "BPM: N/A")
        player.tempo_scale.set(round(other_player.tempo * other_player.tempo_factor / player.tempo, 2))
        print(f"Synchronizacja tempa: {player.name}")

    def load_track(self, player):
        choice = messagebox.askquestion("Wybór", "Chcesz wybrać plik (Yes) czy folder (No)?")
        if choice == 'yes':
            file_path = filedialog.askopenfilename(
                filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")]
            )
            if file_path:
                print(f"Wybrano plik: {file_path}")

                convert_mp3_to_wav(file_path)
        else:
            folder_path = filedialog.askdirectory()
            if folder_path:
                print(f"Wybrano folder: {folder_path}")
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        convert_mp3_to_wav(file_path)
