import csv
import os.path
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf


import omegaconf
from pydub import AudioSegment, silence, effects
from tqdm import tqdm
import itertools

from tokenizer import HebrewTextUtils, TokenizeByLetters
from vall_e.emb.g2p import _get_graphs

from vall_e.emb.qnt import encode_from_file, _replace_file_extension, encode


class Dataset:

    def __init__(self, conf: omegaconf.DictConfig):
        self.name: str = conf.name
        self.wav_path: str = conf.wav_path
        self.metadata_path: str = conf.metadata_path
        self.labeled: str = conf.labeled
        self.length: bool = conf.length


    def create_metadata_csv(self):
        if os.path.isfile(self.metadata_path):
            raise Exception("metadata.csv file exists, are you sure you want to overwrite it?")

        if self.labeled:
            raise Exception(f"dataset is already labeled")

        print(f"Creating {self.metadata_path} for {self.name}")

        import whisper
        model = whisper.load_model("large-v2")

        metadata = open(self.metadata_path, mode='w')
        writer = csv.writer(metadata, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        length = 0

        paths = list(Path(self.wav_path).rglob(f"*.mp3"))
        print(paths)
        for path in tqdm(paths):
            sound = AudioSegment.from_file(path)
            length += sound.duration_seconds
            chunks, timestamp = split_on_silence_with_time_stamps(
                sound,
                min_silence_len=500,
                silence_thresh=sound.dBFS - 16,
                keep_silence=250,  # optional
            )

            for i, chunk in enumerate(chunks):
                np_chunk = pydub_to_np(chunk)
                result = model.transcribe(np_chunk, language='Hebrew')['text']
                time = timestamp[i]
                wav_name = str(path.relative_to(self.wav_path)).replace("/", "_")
                writer.writerow([wav_name, time[0], time[1], result])
                print(f"Transcribed: {wav_name}, {time[0]}, {time[1]}, {result}")

        metadata.close()

        print(f"\nCreated metadata csv file for {self.name} with total length of {length} seconds\n")


    def generate_metadata(self, prepared_data_path: str, model):
        """
         generates metadata.csv in the specified path
         creates .qnt.pt file for every segment
        """
        if self.labeled:
            raise Exception(f"dataset is already labeled")

        if os.path.isfile(self.metadata_path):
            print(f"{self.name} is already labeled, skipping")
            return 0

        print(f"generating metadata for {self.name}")

        metadata = open(self.metadata_path, mode='w')
        writer = csv.writer(metadata, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        audio_paths = list(Path(self.wav_path).rglob(f"*.wav")) + list(Path(self.wav_path).rglob(f"*.mp3"))

        self.length = 0



        for i, path in tqdm(enumerate(audio_paths)):
            sound = AudioSegment.from_file(path)
            sr = sound.frame_rate
            chunks = silence.split_on_silence(
                sound,
                min_silence_len=500,
                silence_thresh=sound.dBFS - 16,
                keep_silence=250,  # optional
            )

            for j, chunk in enumerate(chunks):
                np_chunk = pydub_to_np(chunk)
                self.length += chunk.duration_seconds

                # write to csv
                file_name = f"{i}-{j}"
                result = model.transcribe(np_chunk, language='Hebrew')['text']
                writer.writerow([file_name, result])

                # create .qnt.pt file
                qnt_file_name = f"{self.name}-{file_name}"
                out_path = os.path.join(prepared_data_path, qnt_file_name + ".qnt.pt")
                print(f"{path} - {file_name} - {result}")

                if os.path.isfile(out_path):
                    print(f"Error: qnt path {out_path} already exists")
                    continue

                torch_chunk = torch.from_numpy(np_chunk).unsqueeze(0)
                qnt = encode(torch_chunk, sr, 'cuda')
                torch.save(qnt.cpu(), out_path)

        metadata.close()
        print(f"Generated Metadata for {self.name}")

    def generate_qnt_files(self, prepared_data_path: str):
        if not self.labeled:
            raise Exception(f"dataset not labeled labeled")

        print(f"generating qnt for {self.name}")

        paths = list(Path(self.wav_path).rglob(f"*.wav"))

        for path in tqdm(paths):
            file_name = _replace_file_extension(Path(f"{self.name}-{os.path.basename(path)}"), ".qnt.pt")
            out_path = Path(os.path.join(prepared_data_path, file_name))
            if out_path.exists():
                print("Error: qnt path already exists")
                continue
            qnt = encode_from_file(path)
            torch.save(qnt.cpu(), out_path)

    def generate_normalized_txt_files(self, prepared_data_path: str):
        print(f"creating normalized txt for {self.name}")

        data_frame = pd.read_csv(self.metadata_path, encoding="utf-8", sep='|', header=None)

        for index, row in data_frame.iterrows():
            with open(os.path.join(prepared_data_path, f"{self.name}-{row[0]}.normalized.txt"),
                      'w') as txt_file:
                txt_file.write(
                    HebrewTextUtils.remove_nikud(row[1])
                )
                txt_file.close()

    def __str__(self):
        return f"Dataset - name: {self.name}, length: {self.length}"

    def __repr__(self):
        return self.__str__()


def generate_phoneme_files(prepared_data_path, tokenizer):
    print(f"generating phoneme files")

    paths = list(Path(prepared_data_path).rglob(f"*.normalized.txt"))

    for path in tqdm(paths):
        phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
        if phone_path.exists():
            print("Error: phn path already exists")
            continue

        graphs = _get_graphs(path)
        phones = tokenizer.tokenize(graphs)
        with open(phone_path, "w") as f:
            f.write(" ".join(phones))


def split_on_silence_with_time_stamps(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms

    seek_step - step size for interating over the segment in ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [start - keep_silence, end + keep_silence]
        for (start, end)
        in silence.detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end + next_start) // 2
            range_ii[0] = range_i[1]

    return [
        audio_segment[max(start, 0): min(end, len(audio_segment))]
        for start, end in output_ranges
    ], [(max(start, 0), min(end, len(audio_segment)))
        for start, end in output_ranges]


def pydub_to_np(audio: AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    resampled_audio = audio.set_frame_rate(16000)
    return np.frombuffer(resampled_audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0

if __name__ == "__main__":
    datasets_config = omegaconf.OmegaConf.load("config/saspeech/datasets.yml")

    if sys.argv[1] == "transcribe":
        data_base_name = sys.argv[2]
        print(f"Transcribing {data_base_name}")
        datasets = [Dataset(ds_conf) for ds_conf in datasets_config.datasets]

        for dataset in datasets:
            if dataset.name == data_base_name:
                dataset.create_metadata_csv()


    # datasets_config = omegaconf.OmegaConf.load("config/saspeech/datasets.yml")
    #
    # datasets = [Dataset(ds_conf) for ds_conf in datasets_config.datasets]
    #
    # print("Initialized datasets")
    #
    # model = whisper.load_model("large-v2")
    #
    # for dataset in datasets:
    #     if dataset.labeled:
    #         dataset.generate_qnt_files(datasets_config.prepared_data_path)
    #     else:
    #         dataset.generate_metadata(datasets_config.prepared_data_path, model)
    #
    # for dataset in datasets:
    #     dataset.generate_normalized_txt_files(datasets_config.prepared_data_path)
    #
    #
    # tokenizer = TokenizeByLetters()
    # generate_phoneme_files(datasets_config.prepared_data_path, tokenizer)
    #
    #
    # for dataset in datasets:
    #     print(dataset)


