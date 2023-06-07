import os

import torchaudio
from torch.utils.tensorboard import SummaryWriter

from vall_e.export import main as main_export
from vall_e.__main__ import main as main_test

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():

    _writer = SummaryWriter(log_dir=f"runs/wavs")


    """
        Can only export the current model ckpt due to config file...
    """
    os.system("python -m vall_e.export zoo/saspeech/ar.pt yaml=config/saspeech/ar.yml")
    print(bcolors.OKGREEN + "EXPORTED AR" + bcolors.ENDC)

    os.system("python -m vall_e.export zoo/saspeech/nar.pt yaml=config/saspeech/nar.yml")
    print(bcolors.OKGREEN + "EXPORTED AR" + bcolors.ENDC)

    sentence = "היי, זה משפט לדוגמא כדי שאני אשמע אם המודל מדבר טוב"


    main_test(text=sentence,
         reference="/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/reference.wav",
         out_path=f"/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/test_test.wav",
         ar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/zoo/saspeech/ar.pt",
         nar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/zoo/saspeech/nar.pt",
         device="cuda")

    print(bcolors.OKGREEN + "CREATED WAV" + bcolors.ENDC)


    wav, sr = torchaudio.load(f"/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/test_test.wav")
    print(bcolors.OKGREEN + "LOADED WAV" + bcolors.ENDC)
    _writer.add_audio(tag=f"/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/test_test.wav", snd_tensor=wav, sample_rate=sr)
    print(bcolors.OKGREEN + "SENT TO TENSOR BOARD" + bcolors.ENDC)


    print("SENT TO WRITER")

if __name__ == "__main__":
    main()
