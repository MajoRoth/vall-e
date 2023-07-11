import csv
import os.path
import sys
from datetime import datetime
from pathlib import Path

import numpy
import numpy as np
import pandas as pd
import torch
import torchaudio



import omegaconf
from tqdm import tqdm
import itertools

from tokenizer import HebrewTextUtils, TokenizeByLetters
from vall_e.emb.g2p import _get_graphs

from vall_e.emb.qnt import encode_from_file, _replace_file_extension, encode


class Dataset:

    def __init__(self, conf: omegaconf.DictConfig):
        # self.name: str = conf.name
        # self.wav_path: str = conf.wav_path
        # self.metadata_path: str = conf.metadata_path
        # self.labeled: str = conf.labeled
        # self.length: bool = conf.length

        self.name: str = conf.name
        self.prepared_data_path: str = conf.prepared_data_path
        self.original_wav_path: str = conf.original_wav_path
        self.metadata_path: str = conf.metadata_path
        self.labeled: bool = conf.labeled


    def create_metadata_csv(self, process_number=1, total_process_number=1):
        print(f"Creating metadata for dataset: {self.name}\nprocess {process_number} of total {total_process_number} processes")
        if self.labeled:
            raise Exception(f"dataset is already labeled")

        """
            splitting data for each process
        """
        sub_directories = sorted([x[0] for x in os.walk(self.original_wav_path)])
        process_split = np.array_split(np.array(sub_directories), total_process_number)[process_number - 1]
        print(process_split)

        """
            loading whisper
        """
        import whisper
        model = whisper.load_model("large-v2")


        for directory in process_split:
            """
                load audio
            """
            audio_paths = sorted(Path(directory).glob(f"*.mp3"))

            if len(audio_paths) == 0:
                continue

            print(f"Transcribing {directory}")

            """
                create metadata csv
            """
            directory_relative_path = Path(directory).relative_to(self.original_wav_path)
            absolute_path = Path(self.metadata_path) / directory_relative_path
            csv_absolute_path = absolute_path.with_suffix(".csv")
            csv_absolute_path.parent.mkdir(exist_ok=True, parents=True)
            print(csv_absolute_path)
            metadata = open(csv_absolute_path, mode='w')
            writer = csv.writer(metadata, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            """
                iterate over audio files
            """
            length = 0

            for audio_path in audio_paths:
                result = model.transcribe(str(audio_path), language='Hebrew')
                segments = result["segments"]
                print(audio_path)

                for segment in segments:
                    text = segment['text']
                    start_time = segment['start']
                    end_time = segment['end']
                    index = segment['id']

                    print(f"Transcribed: {str(audio_path)}, {index}, {start_time}, {end_time}, {text}")
                    writer.writerow([str(audio_path), index, start_time, end_time, text])

                length += end_time

            """
                close metadata and write files
            """
            metadata.close()
            print(f"Created metadata csv file for {directory} with total length of {length} seconds!!!\n")

        print("Done")

        # for path in tqdm(process_split):
        #
        #     now = datetime.now()
        #
        #     result = model.transcribe(str(path), language='Hebrew')
        #     segments = result["segments"]
        #
        #     for segment in segments:
        #         text = segment['text']
        #         start_time = segment['start']
        #         end_time =segment['end']
        #         index = segment['id']
        #         print(f"Transcribed: {str(path)}, {index}, {start_time}, {end_time}, {text}")
        #         writer.writerow([str(path), index, start_time, end_time, text])
        #
        #     length += end_time
        #
        #     print(f"Processed {str(path)} in {datetime.now() - now} seconds\ntotal recordings processed: {length / 60} minutes\n")





    def generate_qnt_files(self, process_number=1, total_process_number=1):
        if dataset.labeled:
            self.generate_qnt_files_labled(process_number, total_process_number)
        else:
            self.generate_qnt_files_unlabled(process_number, total_process_number)

    def generate_qnt_files_labled(self, process_number=1, total_process_number=1):
        paths = sorted(Path(self.original_wav_path).rglob(f"*.wav"))
        process_split = np.array_split(np.array(paths), total_process_number)[process_number - 1]

        for path in tqdm(process_split):
            file_name = Path(path).with_suffix(".qnt.pt").name
            qnt_path = (self.prepared_data_path / Path(file_name))

            if qnt_path.exists():
                print(f"Error: qnt path: {qnt_path} already exists")
                continue

            qnt = encode_from_file(path)
            torch.save(qnt.cpu(), qnt_path)

    def generate_qnt_files_unlabled(self, process_number=1, total_process_number=1):
        metadata_paths = sorted(Path(self.metadata_path).rglob(f"*.csv"))
        process_split = np.array_split(np.array(metadata_paths), total_process_number)[process_number - 1]

        for metadata_path in process_split:
            """
                create directory
            """
            drop_csv_suffix = metadata_path.with_suffix("")
            relative_folder_path = drop_csv_suffix.relative_to(self.metadata_path)
            absolute_folder_path = Path(self.prepared_data_path) / relative_folder_path
            absolute_folder_path.mkdir(exist_ok=True, parents=True)
            print(absolute_folder_path)

            """
                iterate over csv
            """
            data_frame = pd.read_csv(metadata_path, encoding="utf-8", sep='|', header=None)

            for index, row in data_frame.iterrows():
                if len(row) == 5:
                    """
                        we have sliced recordings
                    """
                    path, index, start_time, end_time, text = row

                    name_no_suffix = Path(path).with_suffix("").name
                    file_name = f"{name_no_suffix}@{index}"
                    qnt_path = (absolute_folder_path / file_name).with_suffix(".qnt.pt")
                    if qnt_path.exists():
                        print(f"Error: qnt path: {qnt_path} already exists")
                        continue

                    torch_audio, sr = torchaudio.load(path)
                    start_index = int(start_time * sr)
                    end_index = int(end_time * sr)
                    sliced_torch = torch_audio[:, start_index:end_index]

                    try:
                        qnt = encode(sliced_torch, sr, 'cuda')
                        torch.save(qnt.cpu(), qnt_path)
                    except Exception as e:
                        print(f"Couldnt procerss {path}, {index}, {start_time}, {end_time}, {text}")
                        print(f"due to an error {e}")

                else:
                    raise Exception(f"dataset is not in correct format")


    def generate_normalized_txt_files(self, process_number=1, total_process_number=1):
        metadata_paths = sorted(Path(self.metadata_path).rglob(f"*.csv"))
        process_split = np.array_split(np.array(metadata_paths), total_process_number)[process_number - 1]

        for metadata_path in process_split:
            """
                create directory
            """
            drop_csv_suffix = metadata_path.with_suffix("")
            relative_folder_path = drop_csv_suffix.relative_to(self.metadata_path)
            absolute_folder_path = Path(self.prepared_data_path) / relative_folder_path
            absolute_folder_path.mkdir(exist_ok=True, parents=True)
            print(absolute_folder_path)

            """
                iterate over csv
            """
            data_frame = pd.read_csv(metadata_path, encoding="utf-8", sep='|', header=None)

            for index, row in tqdm(data_frame.iterrows()):
                if len(row) == 5:
                    path, index, start_time, end_time, text = row

                    name_no_suffix = Path(path).with_suffix("").name
                    file_name = f"{name_no_suffix}@{index}"
                    normalized_path = (absolute_folder_path / file_name).with_suffix(".normalized.txt")

                    with open(normalized_path, 'w') as txt_file:
                            txt_file.write(
                                HebrewTextUtils.remove_nikud(text)
                            )
                            txt_file.close()

                elif len(row) == 3:
                    path, text, _ = row

                    normalized_path = (absolute_folder_path / Path(path).name).with_suffix(".normalized.txt")

                    with open(normalized_path, 'w') as txt_file:
                            txt_file.write(
                                HebrewTextUtils.remove_nikud(text)
                            )
                            txt_file.close()

                else:
                    raise Exception(f"dataset is not in correct format")

        def generate_phoneme_files(prepared_data_path, tokenizer):
            print(f"generating phoneme files")

            paths = list(Path(prepared_data_path).rglob(f"*.normalized.txt"))

            for path in tqdm(paths):
                phone_path = path.with_suffix(".phn.txt")
                if phone_path.exists():
                    print("Error: phn path already exists")
                    continue

                graphs = _get_graphs(path)
                phones = tokenizer.tokenize(graphs)
                with open(phone_path, "w") as f:
                    f.write(" ".join(phones))


    def generate_phoneme_files(self, tokenizer, process_number=1, total_process_number=1):
        print(f"generating phoneme files")

        paths = sorted(Path(self.prepared_data_path).rglob(f"*.normalized.txt"))
        process_split = np.array_split(np.array(paths), total_process_number)[process_number - 1]

        for path in tqdm(process_split):
            phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
            if phone_path.exists():
                print("Error: phn path already exists")
                continue

            graphs = _get_graphs(path)
            phones = tokenizer.tokenize(graphs)
            with open(phone_path, "w") as f:
                f.write(" ".join(phones))


    def convert_path_to_name_drop_suffix(self, path):
        path = Path(path)
        drop_suffix = path.parent / path.name.split(".")[0]
        if drop_suffix.is_relative_to(self.wav_path):
            return str(drop_suffix.relative_to(self.wav_path)).replace("/", "~")
        else:
            return str(drop_suffix).replace("/", "~")


    def get_file_name(self, path, idx, suffix):
        result = f"{self.name}~{self.convert_path_to_name_drop_suffix(path)}@{int(idx)}.{suffix}"

        return result

    def __str__(self):
        return f"Dataset - name: {self.name}, length: {self.length}"

    def __repr__(self):
        return self.__str__()






if __name__ == "__main__":
    print(f"parameters: {str(sys.argv)}")
    datasets_config = omegaconf.OmegaConf.load("config/hebrew/datasets.yml")
    # datasets_config = omegaconf.OmegaConf.load("config/hebrew/datasets_debug.yml")
    datasets = [Dataset(ds_conf) for ds_conf in datasets_config.datasets]


    if sys.argv[1] == "transcribe":
        data_base_name = sys.argv[2]

        for dataset in datasets:
            if dataset.name == data_base_name:
                if len(sys.argv) > 4:
                    proc_num = int(sys.argv[3])
                    total_num = int(sys.argv[4])
                    dataset.create_metadata_csv(proc_num, total_num)
                else:
                    dataset.create_metadata_csv()

    if sys.argv[1] == "quantize":
        data_base_name = sys.argv[2]
        print(f"Quantizing {data_base_name}")

        for dataset in datasets:
            if dataset.name == data_base_name:
                if len(sys.argv) > 4:
                    proc_num = int(sys.argv[3])
                    total_num = int(sys.argv[4])
                    dataset.generate_qnt_files(proc_num, total_num)
                else:
                    dataset.generate_qnt_files()

    if sys.argv[1] == "normalize":
        data_base_name = sys.argv[2]
        print(f"Normalizing {data_base_name}")

        for dataset in datasets:
            if dataset.name == data_base_name:
                if len(sys.argv) > 4:
                    proc_num = int(sys.argv[3])
                    total_num = int(sys.argv[4])
                    dataset.generate_normalized_txt_files(proc_num, total_num)
                else:
                    dataset.generate_normalized_txt_files()

    if sys.argv[1] == "phoneme":
        data_base_name = sys.argv[2]
        print(f"Normalizing {data_base_name}")

        for dataset in datasets:
            if dataset.name == data_base_name:
                if len(sys.argv) > 4:
                    proc_num = int(sys.argv[3])
                    total_num = int(sys.argv[4])
                    dataset.generate_phoneme_files(TokenizeByLetters(), proc_num, total_num)
                else:
                    dataset.generate_phoneme_files(TokenizeByLetters())






    # for dataset in datasets:
    #     for i in range(3):
    #         dataset.create_metadata_csv(i, 3)

    # print("QNT")
    # for dataset in datasets:
    #     dataset.generate_qnt_files(datasets_config.prepared_data_path)
    #
    # print("TXT")
    # for dataset in datasets:
    #     dataset.generate_normalized_txt_files(datasets_config.prepared_data_path)
    #
    # generate_phoneme_files(datasets_config.prepared_data_path, TokenizeByLetters())

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


