#!/usr/bin/env python3

import os
import pathlib
import shutil
import signal
import sys
import tempfile

import torch
from torch.testing._internal.common_utils import shell

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}

def main():
    for with_init_file in {True, False}:
        tmp_dir = tempfile.mkdtemp()
        init_str = "with {} init_method"
        with_init = init_str.format("file" if with_init_file else "env")
        print(
            "Running distributed tests for the ucc backend {}".format(with_init)
        )
        os.environ["TEMP_DIR"] = tmp_dir
        os.environ["BACKEND"] = "ucc"
        os.environ["INIT_METHOD"] = "env://"
        os.environ["WORLD_SIZE"] = "2" if torch.cuda.device_count() == 2 else "3"

        if with_init_file:
            init_method = f"{tmp_dir}/shared_init_file"
            os.environ["INIT_METHOD"] = init_method
        try:
            os.mkdir(os.path.join(tmp_dir, "barrier"))
            os.mkdir(os.path.join(tmp_dir, "test_dir"))
            command = [sys.executable, "test_ucc.py", "--subprocess"]
            return_code = shell(command)
            if return_code != 0:
                message = f"Failed!"
                if return_code < 0:
                    # subprocess.Popen returns the child process' exit signal as
                    # return code -N, where N is the signal number.
                    signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
                    message += f" Received signal: {signal_name}"
                    raise RuntimeError(message)
        finally:
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
