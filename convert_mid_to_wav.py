from midi2audio import FluidSynth
import soundfile as sf


def midi_to_wav(midi_file, wav_file, soundfont):
    fs = FluidSynth(soundfont)
    fs.midi_to_audio(midi_file, wav_file)


midi_file = 'results/11-10-10-21-08/task2/song_3.mid'
wav_file = 'output.wav'
soundfont_file = "Dore Mark's NY S&S Model B-v5.2.sf2"
midi_to_wav(midi_file, wav_file, soundfont_file)
