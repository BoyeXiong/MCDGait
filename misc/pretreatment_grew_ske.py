import os
import numpy as np
from warnings import warn
from time import sleep
import argparse
import itertools
import threading
import pickle
import time


START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='/GREW', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--subset', default='test/gallery', type=str,
                    help='Subset of GREW, must be one of [train, test/gallery].')
parser.add_argument('--output_path', default='/GREW-ske-pkl', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=8, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
SUBSET = opt.subset
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

print("#"*20)
print(f"SUBSET: {SUBSET}")

if SUBSET not in ['train', 'test/gallery']:
    raise ValueError("subset must be one of [train, test/gallery]")


def get_pickle(thread_id, id_list, save_dir):
    for id in sorted(id_list):
        print(f"Process threadID-PID: {thread_id}-{id}")
        if ".tgz" in id:
            continue
        seq_type = os.listdir(os.path.join(INPUT_PATH, SUBSET, id))
        seq_type.sort()
        for _seq_type in seq_type:
            if ".png" in _seq_type:
                continue
            frame_list = os.listdir(os.path.join(INPUT_PATH, SUBSET, id, _seq_type))
            frame_list.sort()
            pose2d_seq = []
            for frame in frame_list:
                if "2d_pose" not in frame:
                    continue
                frame_path = os.path.join(INPUT_PATH, SUBSET, id, _seq_type, frame)
                f = open(frame_path, "r")
                file = f.read().split(",")
                f.close()
                pose = np.array(file[2:], dtype=np.float32).reshape((-1, 3))
                pose2d_seq.append(pose)
            pose2d_seq = np.asarray(pose2d_seq)
            out_dir = os.path.join(OUTPUT_PATH, SUBSET, id, _seq_type)
            os.makedirs(out_dir)
            pose2d_seq_pkl = os.path.join(out_dir, '{}.pkl'.format(_seq_type))
            pickle.dump(pose2d_seq, open(pose2d_seq_pkl, 'wb'))



if __name__ == '__main__':

    id_list = os.path.join(INPUT_PATH, SUBSET)

    save_dir = opt.output_path

    start_time = time.time()
    maxnum_thread = WORKERS

    all_ids = sorted(os.listdir(id_list))
    num_ids = len(all_ids)

    proces = []
    for thread_id in range(maxnum_thread):
        indices = itertools.islice(range(num_ids), thread_id, num_ids, maxnum_thread)
        id_list = [all_ids[i] for i in indices]   # 多线程id列表
        thread_func = threading.Thread(target=get_pickle, args=(thread_id, id_list, save_dir))

        thread_func.start()
        proces.append(thread_func)

    for proc in proces:
        proc.join()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600,
        (time_elapsed - (time_elapsed // 3600) * 3600) // 60,
        time_elapsed % 60))