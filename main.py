from tkinter import Tk

import track_manager
from gui import AudioMixerGUI
from audio_player import AudioPlayer as ap
from track_manager import *


def main():
    root = Tk()

    track_manager.prepare_music_files()
    player1 = ap("player1")
    player2 = ap("player2")
    player1.load_track(file_path="utils/123.wav")
    player2.load_track(file_path="utils/123.wav")
    player1.change_tempo(1.5)

    app = AudioMixerGUI(root, player1, player2)
    root.mainloop()


if __name__ == "__main__":
    main()
