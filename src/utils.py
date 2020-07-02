import os
import sys
import time
import pickle


def cprint(*s):
    print("\033[1;36m" + " ".join([str(x) for x in s]) + "\033[0;0m", file=sys.stderr)


def save_obj(path: str, obj):
    os.makedirs(path, exist_ok=False)
    with open(os.path.join(path, "data"), "wb") as f:
        pickle.dump(obj, f)


def load_obj(path: str):
    with open(os.path.join(path, "data"), "rb") as f:
        return pickle.load(f)


def trace_num_samples(iterable, step: int):
    start = time.time()
    for idx, elt in enumerate(iterable):
        if idx and (idx % step) == 0:
            cprint("Number of processed elements: ", idx)
            cprint(
                "Number of processed elements per second: ", idx / (time.time() - start)
            )

        yield elt
