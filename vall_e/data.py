import copy
import logging
import random
from collections import defaultdict
from functools import cache, cached_property
from itertools import groupby, zip_longest
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

from .config import cfg
from .sampler import Sampler

"""
    MY CLASS
"""
class ValidCache:
    """
    format is a .txt file:
    path|0
    path|1
    0 stands for invalid
    1 for valid
    """

    def __init__(self, path):
        self.path = path
        self.loaded_data = dict()

        # load cache
        now = time.time()
        print("Loading cache")
        with open(self.path, "r") as txt:
            for line in txt:
                try:
                    file_path, valid = line.split("|")
                    self.loaded_data[str(file_path)] = [False, True][valid]

                except Exception as e:
                    print("Invalid cache line format")
                    continue

        print(f"loaded cache in {time.time() - now} seconds")

    def write_cached(self, file_path: str, value: bool):
        self.loaded_data[str(file_path)] = value

        with open(self.path, "a+") as txt:
            val_to_write = 1 if value else 0
            txt.write(f"{str(file_path)}|{val_to_write}\n")

    def validate(self, path, min_phones, max_phones):
        if str(path) in self.loaded_data:
            print("using cache")
            return self.loaded_data[str(path)]
        else:
            print("calculating and saving")
            result = ValidCache._validate(path, min_phones, max_phones)
            self.write_cached(str(path), result)
            return result



    @staticmethod
    def _validate(path, min_phones, max_phones):
        phones = _get_phones(path)
        unique_phones = list(set(phones))
        if len(unique_phones) == 0:
            return False
        if len(unique_phones) == 1 and unique_phones[0] == "_":
            return False
        if len(phones) < min_phones:
            return False
        if len(phones) > max_phones:
            return False
        return True


torch.multiprocessing.set_sharing_strategy("file_system")

_logger = logging.getLogger(__name__)
_valid_cache = ValidCache(r"/cs/labs/adiyoss/amitroth/vall-e/cache.txt")


def _replace_file_extension(path, suffix):
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


def _get_quant_path(path):
    return _replace_file_extension(path, ".qnt.pt")


def _load_quants(path) -> Tensor:
    """
    Returns:
        quants: (t q)
    """
    path = _get_quant_path(path)
    return torch.load(path)[0].t()


@cache
def _get_phones(path):
    _logger.info(str(path))
    print(path)
    path = _replace_file_extension(path, ".phn.txt")
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
    return ["<s>"] + content.split() + ["</s>"]


def _interleaved_reorder(l, fn):
    groups = defaultdict(list)
    for e in l:
        groups[fn(e)].append(e)
    groups = {k: groups[k] for k in sorted(groups)}
    for interleaved in zip_longest(*groups.values()):
        for value in interleaved:
            if value is not None:
                yield value






@cache
def _validate(path, min_phones, max_phones):
    return _valid_cache.validate(path, min_phones, max_phones)


class VALLEDatset(Dataset):
    def __init__(
        self,
        paths,
        phone_symmap=None,
        spkr_symmap=None,
        min_phones=cfg.min_phones,
        max_phones=cfg.max_phones,
        training=False,
        extra_paths_by_spkr_name: dict[str, list] = {},
    ):
        super().__init__()
        self._head = None
        self.min_phones = min_phones
        self.max_phones = max_phones
        _logger.critical("Validating")
        now = time.time()
        self.paths = [
            path for path in paths if _validate(path, self.min_phones, self.max_phones)
        ]
        _logger.critical(f"Validated data in {time.time() - now}")
        self.spkr_symmap = spkr_symmap or self._get_spkr_symmap()
        self.phone_symmap = phone_symmap or self._get_phone_symmap()
        self.training = training
        _logger.critical("done")

        self.paths_by_spkr_name = self._get_paths_by_spkr_name(extra_paths_by_spkr_name)

        self.paths = [
            p for p in self.paths if len(self.paths_by_spkr_name[cfg.get_spkr(p)]) > 1
        ]

        if len(self.paths) == 0 and training:
            raise ValueError("No valid path is found for training.")

        if training:
            self.sampler = Sampler(self.paths, [cfg.get_spkr])
        else:
            self.sampler = None

    def _get_paths_by_spkr_name(self, extra_paths_by_spkr_name: dict[str, list]):
        ret = defaultdict(list)
        for path in self.paths:
            if _get_quant_path(path).exists():
                ret[cfg.get_spkr(path)].append(path)
        for k, v in extra_paths_by_spkr_name.items():
            ret[k].extend(v)
        return {**ret}

    @cached_property
    def phones(self):
        _logger.critical("--- CALCULATING PHONES --")
        val = sorted(set().union(*[_get_phones(path) for path in self.paths]))
        return val

    def _get_phone_symmap(self):
        # Note that we use phone symmap starting from 1 so that we can safely pad 0.
        return {s: i for i, s in enumerate(self.phones, 1)}

    @cached_property
    def spkrs(self):
        return sorted({cfg.get_spkr(path) for path in self.paths})

    def _get_spkr_symmap(self):
        return {s: i for i, s in enumerate(self.spkrs)}

    def sample_prompts(self, spkr_name, ignore):
        prom_list = []

        choices = set(self.paths_by_spkr_name[spkr_name]) - {ignore}
        choices = [*choices]

        if len(choices) == 0:
            raise ValueError(
                f"Failed to find another different utterance for {spkr_name}."
            )

        for _ in range(cfg.max_prompts):
            path = random.choice(choices)
            prom_list.append(_load_quants(path))
            if random.random() > cfg.p_additional_prompt:
                break

        prom = torch.cat(prom_list)

        return prom

    def __getitem__(self, index):
        if self.training:
            assert self.sampler is not None
            path = self.sampler.sample()
        else:
            path = self.paths[index]

        spkr_name = cfg.get_spkr(path)
        text = torch.tensor([*map(self.phone_symmap.get, _get_phones(path))])
        proms = self.sample_prompts(spkr_name, ignore=path)
        resps = _load_quants(path)
        resp = resps[..., 0]

        return dict(
            path=path,
            spkr_name=spkr_name,
            text=text,
            proms=proms,
            resps=resps,
            resp=resp,
        )

    def head_(self, n):
        self._head = n

    def training_(self, value):
        self.training = value

    def interleaved_reorder_(self, fn):
        self.paths = [*_interleaved_reorder(self.paths, fn)]

    def __len__(self):
        return min(len(self.paths), self._head or len(self.paths))


def collate_fn(samples: list[dict]):
    batch: dict[str, Any] = {k: [s[k] for s in samples] for k in samples[0]}
    return batch


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _create_dataloader(dataset, training):
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size if training else cfg.eval_batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=cfg.nj,
        collate_fn=collate_fn,
        persistent_workers=True,
        worker_init_fn=_seed_worker,
    )


def _load_train_val_paths():
    paths = []
    train_paths = []
    val_paths = []

    for data_dir in cfg.data_dirs:
        paths.extend(tqdm(data_dir.rglob("*.qnt.pt")))

    if len(paths) == 0:
        raise RuntimeError(f"Failed to find any .qnt.pt file in {cfg.data_dirs}.")

    pairs = sorted([(cfg.get_spkr(p), p) for p in paths])

    del paths

    for _, group in groupby(pairs, lambda pair: pair[0]):
        paths = sorted([p for _, p in group])
        random.seed(0)
        random.shuffle(paths)
        n = round(len(paths) * 0.95)
        train_paths.extend(paths[:n])
        val_paths.extend(paths[n:])

    train_paths, val_paths = map(sorted, [train_paths, val_paths])

    return train_paths, val_paths


@cfg.diskcache()
def create_datasets():
    train_paths, val_paths = _load_train_val_paths()

    now = time.time()
    train_dataset = VALLEDatset(
        train_paths,
        training=True,
    )
    _logger.info(f"Created train data set in {time.time() - now}")
    now = time.time()


    val_dataset = VALLEDatset(
        val_paths,
        train_dataset.phone_symmap,
        train_dataset.spkr_symmap,
        extra_paths_by_spkr_name=train_dataset.paths_by_spkr_name,
    )
    _logger.info(f"Created train data set in {time.time() - now}")


    val_dataset.interleaved_reorder_(cfg.get_spkr)
    val_dataset.head_(cfg.max_num_val)

    return train_dataset, val_dataset


def create_train_val_dataloader():
    train_dataset, val_dataset = create_datasets()

    train_dl = _create_dataloader(train_dataset, training=True)
    val_dl = _create_dataloader(val_dataset, training=False)

    _logger.info(str(train_dataset.phone_symmap))
    _logger.info(str(train_dataset.spkr_symmap))

    _logger.info(f"#samples (train): {len(train_dataset)}.")
    _logger.info(f"#samples (val): {len(val_dataset)}.")

    subtrain_dataset = copy.deepcopy(train_dataset)
    subtrain_dataset.interleaved_reorder_(cfg.get_spkr)
    subtrain_dataset.head_(cfg.max_num_val)
    subtrain_dataset.training_(False)
    subtrain_dl = _create_dataloader(subtrain_dataset, training=False)
    assert isinstance(subtrain_dl.dataset, VALLEDatset)

    return train_dl, subtrain_dl, val_dl


if __name__ == "__main__":
    train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
    sample = train_dl.dataset[0]
    print(sample)
