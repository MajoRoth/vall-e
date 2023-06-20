import csv
import os.path
from pathlib import Path

import pandas as pd
import torch
# import whisper
import librosa


import omegaconf
from pydub import AudioSegment, silence
from tqdm import tqdm

from tokenizer import HebrewTextUtils, TokenizeByLetters
from vall_e.emb.g2p import _get_graphs

from vall_e.emb.qnt import encode_from_file, _replace_file_extension, encode



class Dataset:

    def __init__(self, conf: omegaconf.DictConfig):
        self.name: str = conf.name
        self.wav_path: str = conf.wav_path
        self.metadata_path: str = conf.metadata_path
        self.labeled: str = conf.labeled
        self.length: bool = None


    def generate_metadata(self, prepared_data_path: str, model):
        """
         generates metadata.csv in the specified path
         creates .qnt.pt file for every segment
        """
        if self.labeled:
            raise Exception(f"dataset is already labeled")

        if os.path.isfile(self.metadata_path):
            raise Exception(f"metadata file: {self.metadata_path} already exists")

        print("Gnerating metadata")

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
                chunk_array = chunk.get_array_of_samples()
                chunk_tensor = torch.Tensor(chunk_array)

                self.length += chunk.duration_seconds

                # write to csv
                file_name = f"{self.name}-{i}-{j}"

                # result = model.transcribe(chunk_tensor, language='Hebrew')['text']
                result = "בדיקה"
                writer.writerow([file_name, result])

                # create .qnt.pt file
                out_path = os.path.join(prepared_data_path, file_name + ".qnt.pt")
                if os.path.isfile(out_path):
                    print(f"Error: qnt path {out_path} already exists")
                    continue

                print(f"{file_name} - {result}")

                new_chunk_tensor = chunk_tensor.unsqueeze(0)
                qnt = encode(new_chunk_tensor, sr, 'cuda')
                torch.save(qnt.cpu(), out_path)


        metadata.close()
        print(f"Generated Metadata for {self.name}")


    def generate_qnt_files(self, prepared_data_path: str):
        if not self.labeled:
            raise Exception(f"dataset not labeled labeled")

        paths = list(Path(self.wav_path).rglob(f"*.wav"))

        for path in tqdm(paths):
            file_name = _replace_file_extension(Path(f"{self.name}-{os.path.basename(path)}"), ".qnt.pt")
            out_path = Path(os.path.join(prepared_data_path, file_name))
            if out_path.exists():
                print("Error: qnt path already exists")
                continue
            qnt = encode_from_file(path)
            torch.save(qnt.cpu(), out_path)

        print(f"generated wnt for {self.name}")


    def generate_normalized_txt_files(self, prepared_data_path: str):
        data_frame = pd.read_csv(self.metadata_path, encoding="utf-8", sep='|', header=None)

        for index, row in data_frame.iterrows():
            with open(os.path.join(prepared_data_path, f"{row[0]}.normalized.txt"),
                      'w') as txt_file:
                txt_file.write(
                    HebrewTextUtils.remove_nikud(row[1])
                )
                txt_file.close()

        print("created normalized txt")


    def __str__(self):
        return f"Dataset - name: {self.name}, length: {self.length}"

    def __repr__(self):
        return self.__str__()



def generate_phoneme_files(prepared_data_path, tokenizer):
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


if __name__ == "__main__":
    datasets_config = omegaconf.OmegaConf.load("config/saspeech/datasets.yml")

    datasets = [Dataset(ds_conf) for ds_conf in datasets_config.datasets]

    print("Initialized datasets")

    # model = whisper.load_model("large-v2")
    model = None

    # for dataset in datasets:
    #     if dataset.labeled:
    #         dataset.generate_qnt_files(datasets_config.prepared_data_path)
    #     else:
    #         dataset.generate_metadata(datasets_config.prepared_data_path, model)

    # for dataset in datasets:
    #     dataset.generate_normalized_txt_files(datasets_config.prepared_data_path)


    tokenizer = TokenizeByLetters()
    generate_phoneme_files(datasets_config.prepared_data_path, tokenizer)


    for dataset in datasets:
        print(dataset)