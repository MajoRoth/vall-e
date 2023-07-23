import csv
import os.path
import sys
from pathlib import Path
import numpy as np


import omegaconf



class Dataset:
    """
        loads a dataset configurations and initalize a
    """
    def __init__(self, conf: omegaconf.DictConfig):

        self.name: str = conf.name
        self.prepared_data_path: str = conf.prepared_data_path
        self.original_wav_path: str = conf.original_wav_path
        self.metadata_path: str = conf.metadata_path

    def create_metadata_csv(self, process_number: int = 1, total_process_number: int = 1):
        print(f"Creating metadata for dataset: {self.name}\nprocess {process_number} of total {total_process_number} processes")

        # split data if run on multiple processes
        sub_directories = sorted([x[0] for x in os.walk(self.original_wav_path)])
        process_split = np.array_split(np.array(sub_directories), total_process_number)[process_number - 1]
        print(process_split)

        # load whisper
        import whisper
        model = whisper.load_model("large-v2")

        # run over directories and create metadata.csv for each directory
        for directory in process_split:

            audio_paths = sorted(Path(directory).glob(f"*.mp3"))

            if len(audio_paths) == 0:
                continue

            print(f"Transcribing {directory}")

            # create metadata.csv
            directory_relative_path = Path(directory).relative_to(self.original_wav_path)
            absolute_path = Path(self.metadata_path) / directory_relative_path
            csv_absolute_path = absolute_path.with_suffix(".csv")
            csv_absolute_path.parent.mkdir(exist_ok=True, parents=True)
            print(csv_absolute_path)
            metadata = open(csv_absolute_path, mode='w')
            writer = csv.writer(metadata, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # iterate over audio files and transcribe
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

            # close metadata
            metadata.close()
            print(f"Created metadata csv file for {directory}")

        print("Done")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("you should specify database name, process number, and total number of processes as sys args")

    # load yml
    datasets_config = omegaconf.OmegaConf.load("datasets.yml")
    datasets = [Dataset(ds_conf) for ds_conf in datasets_config.datasets]

    data_base_name = sys.argv[1]

    # search datasets and transcribe sleected
    for dataset in datasets:
        if dataset.name == data_base_name:
            if len(sys.argv) > 4:
                proc_num = int(sys.argv[2])
                total_num = int(sys.argv[3])
                dataset.create_metadata_csv(proc_num, total_num)
            else:
                dataset.create_metadata_csv()
