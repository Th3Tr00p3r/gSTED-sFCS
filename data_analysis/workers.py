"""Multiprocessing stuff"""

import multiprocessing as mp
import re
import sys

N_CPU_CORES = mp.cpu_count() // 2  # /2 due to hyperthreading,


def io_worker(
    io_queue,
    data_processing_queue,
    processed_queue,
    data_processor,
    n_files,
    n_processors,
):
    """Doc."""

    print("\n[IO WORKER] Initialized.")
    sys.stdout.flush()
    n_saves = 0
    n_loads = 0
    for func, arg in iter(io_queue.get, "STOP"):
        # func was load task - place processing task in data_processing_queue queue (arg is file_path)
        if func.__name__ == "load_file_dict":  # args is the file_path
            file_idx = int(re.split("_(\\d+)\\.", str(arg))[1])
            file_dict = func(arg)  # load file dict from disk
            byte_data_path = arg.with_name(arg.name.replace(".pkl", "_byte_data.npy"))
            data_processing_queue.put(
                (
                    data_processor.process_data,
                    {
                        "file_idx": file_idx,
                        "full_data": file_dict["full_data"],
                        "byte_data_path": byte_data_path,
                    },
                )
            )
            n_loads += 1
            print("\n[IO WORKER] file loaded from disk, placed in processing queue.")
            sys.stdout.flush()

        # func was save task - no further tasks (args is a 'TDCPhotonFileData' object (p))
        else:
            if arg is not None:
                func(arg)  # raw.dump()
                processed_queue.put(arg)
            n_saves += 1
            print("\n[IO WORKER] processed file saved to disk.")
            sys.stdout.flush()

            if n_saves == n_files:
                # issue as many stop commands as there are processing workers, then stop self
                print("\n[IO WORKER] Saved all files. Stopping processing workers.")
                sys.stdout.flush()
                for _ in range(n_processors):
                    data_processing_queue.put("STOP")
                break

    print(f"\n[IO WORKER] Stopping. Saved {n_saves}/{n_files} files.")
    sys.stdout.flush()


def dump_data_file(p):
    p.raw.dump()


def _get_section_splits(p, **kwargs):
    return p._get_section_splits(**kwargs)


def data_processing_worker(idx: int, data_processing_queue, io_queue, **proc_options):
    """Doc."""

    print(f"\n[WORKER {idx}] Initialized.")
    sys.stdout.flush()
    files_processed = 0
    for func, kwargs in iter(data_processing_queue.get, "STOP"):
        print(f"\n[WORKER {idx}] Processing file {kwargs['file_idx']}")
        sys.stdout.flush()
        p = func(**{**kwargs, **proc_options})
        print(f"\n[WORKER {idx}] Placing processed file in IO Queue")
        sys.stdout.flush()
        io_queue.put((dump_data_file, p))
        files_processed += 1

    print(f"\n[WORKER {idx}] Done! {files_processed} files processed!")
    sys.stdout.flush()
