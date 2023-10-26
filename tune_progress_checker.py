import os, datetime, time, glob
from tune_hyperparams import load_result


def check():
    for d, sd, f in os.walk("models/RayTune"):
        if len(d) == len("models/RayTune/tune_20hz_Fx"):
            total_trials = []
            for sd_ in sd:
                for dir, subdir, files in os.walk(f"{d}/{sd_}"):
                    if not subdir and files:
                        total_trials.append(sd_)
            num_errored_trials = len(glob.glob(f"{d}/**/error.txt", recursive=True))
            try:
                num_finished_trials = load_result(
                    d, print_logs=False, return_success=True
                )
            except:
                num_finished_trials = None
                print(f"{d}: Result not prepared yet.")
            print(f"{d}: {num_finished_trials}/300 ({num_errored_trials} errored)")


def start_checking():
    start_time = datetime.datetime.now()
    while True:
        new_time = datetime.datetime.now()
        check()
        print(f"Elapsed time: {new_time - start_time}\n")
        time.sleep(5)


if __name__ == "__main__":
    start_checking()
