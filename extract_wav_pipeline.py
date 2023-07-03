import os
import time
from datetime import datetime

import torchaudio
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from vall_e.export import main as main_export
from vall_e.__main__ import main as main_test




def main():
    now = datetime.now()
    FORMAT = "%d-%m-%Y-%H:%M"
    _writer = SummaryWriter(log_dir=f"runs/wavs_{now.strftime(FORMAT)}")

    yaml_cfg = OmegaConf.load("config/hebrew/config.yml")

    latest_ar = None
    new_ar_ckpt = False
    latest_nar = None
    new_nar_ckpt = False

    while True:
        if os.path.isfile(yaml_cfg.latest_ar_path):  # Check for new ar ckpt
            f = open(yaml_cfg.latest_ar_path, 'r')
            latest_ar_current_value = f.read()

            if latest_ar_current_value != latest_ar:
                os.system(f"python -m vall_e.export {yaml_cfg.ar_ckpt_path} yaml=config/hebrew/ar.yml")
                print(" ")
                print(f"EXPORTED AR {latest_ar_current_value}")
                print(" ")

                latest_ar = latest_ar_current_value
                new_ar_ckpt = True
        else:
            print(" ")
            print(f"path does not exist {yaml_cfg.latest_ar_path}")
            print(" ")



        if os.path.isfile(yaml_cfg.latest_nar_path):  # Check for new nar ckpt
            f = open(yaml_cfg.latest_nar_path, 'r')
            latest_nar_current_value = f.read()

            if latest_nar_current_value != latest_nar:
                os.system(f"python -m vall_e.export {yaml_cfg.nar_ckpt_path} yaml=config/hebrew/nar.yml")
                print(" ")
                print(f"EXPORTED NAR {latest_nar_current_value}")
                print(" ")

                latest_nar = latest_nar_current_value
                new_nar_ckpt = True
        else:
            print(" ")

            print(f"path does not exist {yaml_cfg.latest_ar_path}")
            print(" ")



        if new_ar_ckpt and new_nar_ckpt:  # Create wav from 2 new ckpts
            try:
                ckpt_name = latest_ar.split('_')[1]
            except Exception as e:
                print(f"\nCouldnt deine ckpt_name. latest_ar ckpt is {latest_ar}.\n")
                ckpt_name = 'default'


            print(f"new ckpt {ckpt_name}")
            new_ar_ckpt = False
            new_nar_ckpt = False

            sentence = "היי, זה משפט לדוגמא כדי שאני אשמע אם המודל מדבר טוב"

            for ref in [yaml_cfg.reference_path_hayot, yaml_cfg.reference_path_amit, yaml_cfg.reference_path_shahar]:
                file_name = ref.split("/")[-1]
                file_name_without_wav = file_name.split(".")[0]
                wav_name = f"test_{file_name_without_wav}_{ckpt_name}.wav"
                output_path = f"{yaml_cfg.output_path}/{wav_name}"

                main_test(text=sentence,
                          reference=yaml_cfg.reference_path_hayot,
                          out_path=output_path,
                          ar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/zoo/hebrew/ar.pt",
                          nar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/zoo/hebrew/nar.pt",
                          device="cuda")


                print(f"CREATED WAV test_{file_name_without_wav}_{ckpt_name}")
                wav, sr = torchaudio.load(output_path)

                print(f"LOADED WAV {wav_name}")
                _writer.add_audio(tag=f"{wav_name}", snd_tensor=wav,
                                  sample_rate=sr)

                print("SENT TO TENSOR BOARD")






if __name__ == "__main__":
    main()
