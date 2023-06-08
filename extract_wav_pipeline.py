import os
from datetime import datetime

import torchaudio
from torch.utils.tensorboard import SummaryWriter

from vall_e.export import main as main_export
from vall_e.__main__ import main as main_test

LATEST_PATH_AR = "/cs/labds/adiyoss/amitroth/vall-e/ckpts/saspeech/ar/model/lates"
LATEST_PATH_NAR = "/cs/labds/adiyoss/amitroth/vall-e/ckpts/saspeech/"


def main():
    now = datetime.now()
    FORMAT = "%d-%m-%Y-%H:%M"
    _writer = SummaryWriter(log_dir=f"runs/wavs_{now.strftime(FORMAT)}")

    latest_ar = None
    new_ar_ckpt = False
    latest_nar = None
    new_nar_ckpt = False


    while True:

        if os.path.isfile(LATEST_PATH_AR):  # Check for new ar ckpt
            f = open(LATEST_PATH_AR, 'r')
            latest_ar_current_value = f.read()

            if latest_ar_current_value != latest_ar:
                os.system("python -m vall_e.export zoo/saspeech/ar.pt yaml=config/saspeech/ar.yml")
                print(f"EXPORTED AR {latest_ar_current_value}")

                latest_ar = latest_ar_current_value
                new_ar_ckpt = True

        if os.path.isfile(LATEST_PATH_NAR):  # Check for new nar ckpt
            f = open(LATEST_PATH_NAR, 'r')
            latest_nar_current_value = f.read()

            if latest_nar_current_value != latest_nar:
                os.system("python -m vall_e.export zoo/saspeech/nar.pt yaml=config/saspeech/nar.yml")
                print(f"EXPORTED NAR {latest_nar_current_value}")

                latest_nar = latest_nar_current_value
                new_nar_ckpt = True


        if new_ar_ckpt and new_nar_ckpt:  # Create wav from 2 new ckpts
            new_ar_ckpt = False
            new_nar_ckpt = False

            try:
                ckpt_number = int(latest_ar.split('_')[1])
            except Exception as e:
                print(e)
                ckpt_number = 99999

            sentence = "היי, זה משפט לדוגמא כדי שאני אשמע אם המודל מדבר טוב"

            main_test(text=sentence,
                      reference="/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/reference.wav",
                      out_path=f"/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/test_{ckpt_number}.wav",
                      ar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/zoo/saspeech/ar.pt",
                      nar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/zoo/saspeech/nar.pt",
                      device="cuda")

            print("CREATED WAV test_{ckpt_number}")
            wav, sr = torchaudio.load(f"/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/test_{ckpt_number}.wav")
            print("LOADED WAV test_{ckpt_number}")
            _writer.add_audio(tag=f"/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/test_{ckpt_number}.wav", snd_tensor=wav,
                              sample_rate=sr)
            print("SENT TO TENSOR BOARD")






if __name__ == "__main__":
    main()
