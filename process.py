import copy
import json
import pickle
import re
from pathlib import Path

import miditoolkit
import numpy as np
from tqdm import tqdm


DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

DEFAULT_RESOLUTION = 480


class Item:
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return f"Item(name={self.name}, start={self.start}, end={self.end}, velocity={self.velocity}, pitch={self.pitch})"


def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))

    for note in notes:
        note_items.append(
            Item(
                name='Note', 
                start=note.start, 
                end=note.end, 
                velocity=note.velocity, 
                pitch=note.pitch
            )
        )

    note_items.sort(key=lambda x: x.start)

    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(
            Item(
                name='Tempo',
                start=tempo.time,
                end=None,
                velocity=None,
                pitch=int(tempo.tempo)
            )
        )

    tempo_items.sort(key=lambda x: x.start)

    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)

    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(
                Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick]
                )
            )

        else:
            output.append(
                Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=output[-1].pitch
                )
            )

    tempo_items = output
    return note_items, tempo_items


def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift

    return items      


# extract chord
def chord_extract(midi_path, max_time):
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    interval = midi_obj.ticks_per_beat * 1
    chord_items = []
    ###
    # implement your chord extraction here
    # it's fine if you don't use chord items
    ###
    return chord_items


# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []

    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)

    return groups


class Event:
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f'Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})'

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue

        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(
            Event(
                name='Bar',
                time=None, 
                value=None,
                text=str(n_downbeat)
            )
        )

        for item in groups[i][1:-1]:
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(
                Event(
                    name='Position', 
                    time=item.start,
                    value=f'{index+1}/{DEFAULT_FRACTION}',
                    text=str(item.start)
                )
            )

            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right'
                ) - 1

                events.append(
                    Event(
                        name='Note Velocity',
                        time=item.start, 
                        value=velocity_index,
                        text=f'{item.velocity}/{DEFAULT_VELOCITY_BINS[velocity_index]}'
                    )
                )

                # pitch
                events.append(
                    Event(
                        name='Note On',
                        time=item.start, 
                        value=item.pitch,
                        text=str(item.pitch)
                    )
                )

                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))

            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))

            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)

                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)     
    return events


def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events


def write_midi(words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []

    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note Velocity' and \
            events[i+2].name == 'Note On' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
            index = int(events[i+1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+2].value)
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == 'Position' and events[i+1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i+1].value])
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Tempo Class' and \
            events[i+2].name == 'Tempo Value':
            position = int(events[i].value.split('/')[0]) - 1
            if events[i+1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+2].value)
            elif events[i+1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+2].value)
            elif events[i+1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+2].value)
            temp_tempos.append([position, tempo])

    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))

    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])

    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])

    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)


def find_deepest_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory")

    max_depth = -1
    deepest_files = []

    def recurse(path, depth):
        nonlocal max_depth, deepest_files
        has_files = False
        for entry in path.iterdir():
            if entry.is_dir():
                recurse(entry, depth + 1)
            elif entry.is_file():
                has_files = True
                if depth > max_depth:
                    max_depth = depth
                    deepest_files = [entry]
                elif depth == max_depth:
                    deepest_files.append(entry)

    recurse(directory, 0)
    return sorted([str(f) for f in deepest_files], key=lambda x: int(re.search(r'(\d+)\.mid$', x).group(1)))


if __name__ == '__main__':
    directory_path = "Pop1K7/midi_analyzed"
    file_paths = find_deepest_files(directory_path)
    event2word, word2event = pickle.load(open('./basic_event_dictionary.pkl', 'rb'))

    data = []
    for path in tqdm(file_paths):
        midi = miditoolkit.midi.parser.MidiFile(path)
        note_items, tempo_items = read_items(path)
        note_items = quantize_items(note_items)
        chord_items = chord_extract(path, note_items[-1].end)

        # if using chord items
        # items = chord_items + tempo_items + note_items
        # if not using chord items
        items = tempo_items + note_items

        max_time = note_items[-1].end
        groups = group_items(items, max_time)
        events = item2event(groups)

        words = []
        for event in events:
            e = f'{event.name}_{event.value}'

            if e in event2word:
                words.append(event2word[e])
            else:
                # OOV
                if event.name == 'Note Velocity':
                    # replace with max velocity based on our training data
                    words.append(event2word['Note Velocity_21'])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print('something is wrong! {}'.format(e))

        data.append(
            {
                "path": path,
                "tokens": words,
            }
        )

    with open("data.json", "w") as f:
        json.dump(data, f)
