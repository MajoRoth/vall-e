import csv
import os
import shutil
from enum import Enum
from typing import List, Tuple, Generic, Type
from pathlib import Path
import whisper
from pydub import AudioSegment


import pandas as pd
import torch
from pydub.silence import split_on_silence
from tqdm import tqdm

from tokenizer import Tokenizer, TokenizeByLetters, HebrewTextUtils
from vall_e.emb.g2p import _get_graphs
from vall_e.emb.qnt import _replace_file_extension, encode_from_file


class Dataset:
    def __init__(self, name: str, absolute_path: str,
                 metadata_path: str = "metadata.csv",
                 wav_path: str = ""):

        self.wav_path = wav_path
        self.metadata_path = metadata_path
        self.absolute_path = absolute_path
        self.name = name

        self.absolute_metadata_path = os.path.join(self.absolute_path, self.metadata_path)
        self.absolute_wav_path = os.path.join(self.absolute_path, self.wav_path)

        self.length = None

        # if not os.path.exists(self.absolute_wav_path):
        #     raise Exception(f"wav folder path: {self.absolute_wav_path} is invalid")

    def create_metadata_from_wav(self):
        # if os.path.isfile(self.absolute_metadata_path):
        #     raise Exception("metadata.csv file exists, are you sure you want to overwrite it?")

        model = whisper.load_model("large-v2")

        # metadata = open(self.absolute_metadata_path + "_debug", mode='w')
        # writer = csv.writer(metadata, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        paths = list(Path(self.absolute_wav_path).rglob(f"*.wav"))
        for path in tqdm(paths):
            result = model.transcribe(str(path), language='Hebrew')['text']
            print(f"{os.path.split(path)[1]} - {result}")
            # writer.writerow([os.path.split(path)[1], result])

        # metadata.close()
        print("\nDone\n")


class Mp3Dataset(Dataset):
    """
        Suitable for datasets which contains only folders of mp3 files
    """

    def __init__(self, name: str, absolute_path: str, mp3_files_absolute_path: str, metadata_path: str = "metadata.csv", wav_path: str = ""):
        # if not os.path.exists(absolute_path):
            # create folder of the dataset
            # os.makedirs(absolute_path)


        # if not os.path.exists(os.path.join(absolute_path, wav_path)):
            # create folder of the dataset
            # os.makedirs(os.path.join(absolute_path, wav_path))

        super().__init__(name, absolute_path, metadata_path, wav_path)
        self.mp3_files_absolute_path = mp3_files_absolute_path

    def orgainze_dataset(self):
        """
            orgainze all of the mp3 files into 1 folder
            convert mp3 to wav
            create metadata csv file
        """


        mp3_paths = list(Path(self.mp3_files_absolute_path).rglob(f"*.mp3"))
        for i, path in tqdm(enumerate(mp3_paths)):
            sound = AudioSegment.from_mp3(path)
            print(sound.duration_seconds)
            print(type(sound))
            # sound.export(
            #     os.path.join(self.absolute_wav_path, f"{self.name}-{i}.wav"),
            #     format="wav")

            print(f"exported {path}")

        print("finished")

        self.create_metadata_from_wav()


    def organize_if_needed(self):
        """
            call organized data set only if cant find the preprocessed wavs
        """
        if not os.path.exists(self.absolute_path):
            # create folder of the dataset
            os.makedirs(self.absolute_path)
            os.makedirs(self.absolute_wav_path)
            self.organize_dataset()
        elif not os.path.exists(self.absolute_wav_path):
            os.makedirs(self.absolute_wav_path)
            self.organize_dataset()

        else:
            print(f"{self.name} is already organized")



class PrepareData:
    """
        Specify folders of raw data and exectue preprocess in order to train vall-e

        to train vall-e we need to specify a folder with all of the wav files,
        and for every wav file append additional 3 files:
        1. file.wav
        2. file.normalized.txt
        3. file.phn.txt
        4 file.qnt.pt


        the pipe line is like so:
        1. if data is not tagged - create a metadata.csv using asr or by other methods
        2. after the data is tagged - make sure all of the wav files are in same folder and there is a metadata.csv file
        3. specify all of the folders you want to include for training
        4. move the folders into the main folder
        5. create file.normalized.txt using this class
        6. create file.phn.txt and file.qnt.pt using vall-e's bult in scripts
    """

    """
        data_folders_list = List of tuples containing metadata.csv path, and wav folder path.
    """

    def __init__(self, processed_data_absolute_path: str, datasets_list: List[Dataset],
                 tokenizer: Type[Tokenizer] = TokenizeByLetters):
        self.processed_data_absolute_path = processed_data_absolute_path
        self.dataset_list = dataset_list
        self.tokenizer = tokenizer

        for dataset in self.dataset_list:
            if isinstance(dataset, Mp3Dataset):
                dataset.organize_if_needed()
            print(f"Dataset: {dataset.name}, length: {dataset.length} seconds")

    def move_wav_files(self):
        print("--- Moving Wav Files ---")
        for dataset in self.dataset_list:
            print(f"Moving files from {dataset.absolute_wav_path}")
            paths = list(Path(dataset.absolute_wav_path).rglob(f"*.wav")) + list(Path(dataset.absolute_wav_path).rglob(f"*.mp3"))

            for path in tqdm(paths):
                file_name = os.path.split(path)[1]
                shutil.copyfile(path, os.path.join(self.processed_data_absolute_path, f"{dataset.name}-{file_name}"))

    def create_normalized_txt_files(self):
        print("Creating Normalized Text Files")
        for dataset in self.dataset_list:
            data_frame = pd.read_csv(dataset.absolute_metadata_path, encoding="utf-8", sep='|', header=None)
            data_folder_name = os.path.split(os.path.split(dataset.absolute_metadata_path[0])[0])[1]
            print(f"Tokenizing {data_folder_name}")

            for index, row in data_frame.iterrows():
                with open(os.path.join(self.processed_data_absolute_path, f"{dataset.name}-{row[0]}.normalized.txt"),
                          'w') as txt_file:
                    txt_file.write(
                        HebrewTextUtils.remove_nikud(row[1])
                    )
                    txt_file.close()

    def create_phoneme_files(self):
        paths = list(Path(self.processed_data_absolute_path).rglob(f"*.normalized.txt"))

        for path in tqdm(paths):
            phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
            if phone_path.exists():
                print("Error: phn path already exists")
                continue

            graphs = _get_graphs(path)
            phones = self.tokenizer.tokenize(graphs)
            with open(phone_path, "w") as f:
                f.write(" ".join(phones))

    def create_qnt_files(self):
        print("quant")
        paths = list(Path(self.processed_data_absolute_path).rglob(f"*.wav"))

        for path in tqdm(paths):
            out_path = _replace_file_extension(path, ".qnt.pt")
            if out_path.exists():
                print("Error: qnt path already exists")
                continue
            qnt = encode_from_file(path)
            torch.save(qnt.cpu(), out_path)

    def execute_in_series(self):
        self.move_wav_files()
        self.create_normalized_txt_files()
        self.create_qnt_files()  # Todo debug in cluster
        self.create_phoneme_files()  # Todo debug in cluster



if __name__ == "__main__":
    # datasets_list = [
    #     Dataset(name="hayot-kis", absolute_path="/Users/amitroth/Data/hayot_kis/saspeech_gold_standard",
    #             metadata_path="metadata.csv", wav_path="wavs_24k/"),
    #
    #     Mp3Dataset(name="osim-history",
    #                       absolute_path="/cs/dataset/Download/adiyoss/heb_data/osim-history/",
    #                       mp3_files_absolute_path="/cs/dataset/Download/adiyoss/podcasts/parsed_data/osim-history",
    #                       metadata_path="metadata.csv"),
    #
    #     Mp3Dataset(name="geekonomy",
    #                absolute_path="/cs/dataset/Download/adiyoss/heb_data/geekonomy/",
    #                mp3_files_absolute_path="/cs/dataset/Download/adiyoss/podcasts/parsed_data/geekonomy",
    #                metadata_path="metadata.csv")
    # ]

    dataset_list = [
        Dataset(name="podcasts",
                   absolute_path="/Users/amitroth/Data/hayot_kis/saspeech_gold_standard/", wav_path="wavs_24k",
                   metadata_path="metadata.csv")
    ]

    # Make sure before preparing the data the every folder has a metadata.csv file, and if not, create using Whisper.
    prepare_data = PrepareData(
        processed_data_absolute_path="/Users/amitroth/Data/ready",
        datasets_list=dataset_list,
        tokenizer=TokenizeByLetters
    )

    prepare_data.execute_in_series()

    # dataset = Dataset(name="hayot-kis",
    #                   absolute_path="/cs/dataset/Download/adiyoss/podcasts/hayot_kis/saspeech_gold_standard",
    #                   metadata_path="metadata.csv", wav_path="wavs_24k/")

    # mp3_dataset = Mp3Dataset(name="podcasts",
    #                   absolute_path="/cs/dataset/Download/adiyoss/heb_data/podcasts/",
    #                   mp3_files_absolute_path="/cs/dataset/Download/adiyoss/podcasts/parsed_data/osim-history",
    #                   metadata_path="metadata.csv")

    # mp3_dataset = Mp3Dataset(name="podcasts",
    #                          absolute_path="/Users/amitroth/Data/mp3_data_processed",
    #                          mp3_files_absolute_path="/Users/amitroth/Data/mp3_data",
    #                          metadata_path="metadata.csv")

