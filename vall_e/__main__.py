import argparse
from pathlib import Path

import torch
from einops import rearrange

from .emb import g2p, qnt
from .utils import to_device


def main(text=None, reference=None, out_path=None, ar_ckpt=None, nar_ckpt=None, device=None):
    if text is None:
        parser = argparse.ArgumentParser("VALL-E TTS")
        parser.add_argument("text")
        parser.add_argument("reference", type=Path)
        parser.add_argument("out_path", type=Path)
        parser.add_argument("--ar-ckpt", type=Path, default="zoo/ar.pt")
        parser.add_argument("--nar-ckpt", type=Path, default="zoo/nar.pt")
        parser.add_argument("--device", default="cuda")
        args = parser.parse_args()

        text = args.text
        reference = args.reference
        out_path = args.out_path
        ar_ckpt = args.ar_ckpt
        nar_ckpt = args.nar_ckpt
        device = args.device


    ar = torch.load(ar_ckpt).to(device)
    nar = torch.load(nar_ckpt).to(device)

    symmap = ar.phone_symmap

    proms = qnt.encode_from_file(reference)
    proms = rearrange(proms, "1 l t -> t l")

    phns = torch.tensor([symmap[p] for p in g2p.encode(text)])

    proms = to_device(proms, device)
    phns = to_device(phns, device)

    resp_list = ar(text_list=[phns], proms_list=[proms])
    resps_list = [r.unsqueeze(-1) for r in resp_list]

    resps_list = nar(text_list=[phns], proms_list=[proms], resps_list=resps_list)
    qnt.decode_to_file(resps=resps_list[0], path=out_path)
    print(out_path, "saved.")


if __name__ == "__main__":
    main(text="אני אוהב מאוד לאכול מעדן תות",
         reference="/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/reference.wav",
         out_path="/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/out1.wav",
         ar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/ckpts/saspeech/ar/model/default/mp_rank_00_model_states.pt",
         nar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/ckpts/saspeech/nar/model/default/mp_rank_00_model_states.pt",
         device="cuda")

    main(text="אני עכשיו משלם אצל המוכרת בסופר",
         reference="/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/reference.wav",
         out_path="/cs/labs/adiyoss/amitroth/vall-e/output/saspeech/out2.wav",
         ar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/ckpts/saspeech/ar/model/default/mp_rank_00_model_states.pt",
         nar_ckpt="/cs/labs/adiyoss/amitroth/vall-e/ckpts/saspeech/nar/model/default/mp_rank_00_model_states.pt",
         device="cuda")
