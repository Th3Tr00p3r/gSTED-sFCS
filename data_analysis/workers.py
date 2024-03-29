"""Multiprocessing stuff"""

import multiprocessing as mp
import re

N_CPU_CORES = mp.cpu_count() // 2  # /2 due to hyperthreading,


def io_worker(
    io_queue,
    data_processing_queue,
    processed_queue,
    data_processor,
    n_files,
    n_processors,
):
    n_saves = 0
    n_loads = 0
    #    print("IO WORKER: INITIALIZED.")  # TESTESTEST
    for func, arg in iter(io_queue.get, "STOP"):
        #        print("IO WORKER: LOADING DATA... ", end="")  # TESTESTEST
        # func was load task - place processing task in data_processing_queue queue
        # arg is file_path
        if func.__name__ == "load_file_dict":  # args is the file_path
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
        # func was save task - no further tasks
        else:  # args is a 'TDCPhotonFileData' object (p)
            if arg is not None:
                #                print(f"SAVING RAW DATA {arg.idx}... ", end="")  # TESTESTEST
                func(arg)  # raw.dump()
                processed_queue.put(arg)
                #                print("Done!") # TESTESTEST
            n_saves += 1
            #            print(f"saved {n_saves} files!")  # TESTESTEST
            if n_saves == n_files:
                # issue as many stop commands as there are processing workers, then stop self
                for _ in range(n_processors):
                    data_processing_queue.put("STOP")
                #                print("STOPPING IO WORKER!")  # TESTESTEST
                #                io_queue.put("STOP")
                break


#    print(f"IO WORKER: TERMINATED! SAVED {n_saves}/{n_files} FILES.")  # TESTESTEST


def dump_data_file(p):
    p.raw.dump()


def _get_section_splits(p, **kwargs):
    return p._get_section_splits(**kwargs)


def data_processing_worker(data_processing_queue, io_queue, **proc_options):
    #    print("PROCESSING WORKER: INITIALIZED.")  # TESTESTEST
    files_processed = 0  # TESTESTEST
    for func, args, kwargs in iter(data_processing_queue.get, "STOP"):
        #        print(F"\nPROCESSING WORKER: args={args}, kwargs={kwargs.keys()}\n") # TESTESTEST
        #        print("PROCESSING WORKER: PROCESSING... ", end="")  # TESTESTEST
        p = func(*args, **{**kwargs, **proc_options})
        #        print(f"FILE {p.idx} PROCESSED.")  # TESTESTEST
        io_queue.put((dump_data_file, p))
        files_processed += 1


#    print(f"PROCESSING WORKER: TERMINATED! PROCESSED {files_processed} FILES.")  # TESTESTEST
