"""Multiprocessing stuff"""

import re
import sys
from datetime import datetime

import psutil

N_CPU_CORES = psutil.cpu_count(logical=False)


def io_worker(
    io_queue,
    data_processing_queue,
    processed_list,
    data_processor,
    n_files,
    n_processors,
):
    """Doc."""

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [IO WORKER] Initialized.")
    sys.stdout.flush()
    n_saves = 0
    n_loads = 0
    for func, arg in iter(io_queue.get, "STOP"):
        if func.__name__ == "load_file_dict":
            file_idx = int(re.split("_(\\d+)\\.", str(arg))[1])
            file_dict = func(arg)
            byte_data_path = arg.with_name(arg.name.replace(".pkl", "_byte_data.npy"))
            data_processing_queue.put(
                (
                    data_processor.process_data,
                    (file_idx, file_dict["full_data"]),
                    {"byte_data_path": byte_data_path},
                )
            )
            n_loads += 1
            print(
                f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [IO WORKER] file loaded from disk, placed in processing queue."
            )
            sys.stdout.flush()
        else:
            if arg is not None:
                func(arg)  # raw.dump()
                print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [IO WORKER] {arg} dumped to disk.")
                sys.stdout.flush()
                processed_list.append(arg)
            n_saves += 1

            if n_saves == n_files:
                print(
                    f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [IO WORKER] Dumped all files. Stopping all processer workers."
                )
                sys.stdout.flush()
                for _ in range(n_processors):
                    data_processing_queue.put("STOP")
                break

    print(
        f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [IO WORKER] Stopped. Saved {n_saves}/{n_files} files."
    )
    sys.stdout.flush()


def dump_data_file(p):
    p.raw.dump()


def _get_section_splits(p, **kwargs):
    return p._get_section_splits(**kwargs)


def data_processing_worker(idx: int, data_processing_queue, io_queue, **proc_options):
    """Doc."""

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [WORKER {idx}] Initialized.")
    sys.stdout.flush()
    files_processed = 0
    for func, args, kwargs in iter(data_processing_queue.get, "STOP"):
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [WORKER {idx}] Processing file {args[0]}")
        sys.stdout.flush()
        try:
            p = func(*args, **{**kwargs, **proc_options})
            if p is None:
                raise RuntimeError("Processing returned 'None'")
        except Exception as exc:
            p = None
            print(
                f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [WORKER {idx}] Error encountered: {exc}. Placing 'None' in IO Queue."
            )
            sys.stdout.flush()
        else:
            print(
                f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [WORKER {idx}] Placing processed file {p.file_num} in IO Queue"
            )
            sys.stdout.flush()
        io_queue.put((dump_data_file, p))
        files_processed += 1

    print(
        f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] [WORKER {idx}] Done! {files_processed} files processed!"
    )
    sys.stdout.flush()
