import time
import pathlib
import logging

logging.basicConfig(level=logging.DEBUG)


def get_paths(path):
    answer = {}
    for x in pathlib.Path(path).rglob("*"):
        try:
            answer[str(x)] = (x.stat().st_ctime, x.is_dir())
        except FileNotFoundError:
            pass
    return answer


def log(name, is_dir, action, run_time=""):
    descrip = "Directory" if is_dir else "File"
    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info("{} {} {} {}: {}".format(curr_time, run_time, descrip, action, name))


def scan(top_dir, sleep_time):
    old_paths = get_paths(top_dir)
    s_old_paths = set(old_paths)

    while True:
        time.sleep(sleep_time)
        new_paths = get_paths(top_dir)
        s_new_paths = set(new_paths)
        cre_names = s_new_paths - s_old_paths
        del_names = s_old_paths - s_new_paths

        for name in cre_names:
            _, is_dir = new_paths[name]
            created_time = int(time.time_ns() / 1000000)
            # log(name, is_dir, "created")

        for name in del_names:
            _, is_dir = old_paths[name]
            time_taken = int(time.time_ns() / 1000000) - created_time
            log(name, is_dir, "deleted", run_time=str(time_taken))

        # for name in s_old_paths & s_new_paths:
        #     new_time, is_dir = new_paths[name]
        #     old_time, _ = old_paths[name]
        #     if new_time != old_time:
        #         log(name, is_dir, "modified")

        old_paths = new_paths
        s_old_paths = s_new_paths


top_dir = "/scratch/dkarkalousos/tmp/"
sleep_time = 0
scan(top_dir, sleep_time)