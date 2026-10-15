import time
import subprocess
import os
import datetime
import argparse
import signal

POLL_INTERVAL = 3


def parse_args():
    parser = argparse.ArgumentParser(description="GPU Worker")
    parser.add_argument(
        "--cmd_file",
        type=str,
        required=True,
        help="Path to commands file"
    )
    return parser.parse_args()


def read_commands(command_file):
    with open(command_file, "r") as f:
        return f.readlines()


def write_commands(command_file, lines):
    with open(command_file, "w") as f:
        f.writelines(lines)


def fetch_next_command(command_file):
    lines = read_commands(command_file)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped != "STOP":
            return idx, stripped, lines

    return None, None, lines


def mark_done(command_file, idx, lines):
    lines[idx] = "# " + lines[idx]
    write_commands(command_file, lines)


def should_stop(command_file):
    with open(command_file, "r") as f:
        for line in f:
            if line.strip() == "STOP":
                return True
    return False


def clear_stop(command_file):
    lines = read_commands(command_file)
    lines = [l for l in lines if l.strip() != "STOP"]
    write_commands(command_file, lines)


def start_holder():
    print("[Worker] Start holder")
    return subprocess.Popen(
        ["python", "holder.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )


def stop_proc(proc):
    if proc is None:
        return

    if proc.poll() is None:
        print("[Worker] Killing process group...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def run_task(cmd):
    print(f"[Worker] Running task: {cmd}")

    os.makedirs("logs", exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = cmd.replace(" ", "_").replace("/", "")[:50]
    log_file = f"logs/{safe_name}_{ts}.log"

    log_f = open(log_file, "w")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid  # 👈 关键：进程组
    )

    return proc, log_f, log_file


def main():
    args = parse_args()
    command_file = args.cmd_file

    print(f"[Worker] Using command file: {command_file}")

    holder = start_holder()

    while True:
        idx, cmd, lines = fetch_next_command(command_file)

        if cmd:
            print("[Worker] Task detected")

            stop_proc(holder)

            proc, log_f, log_file = run_task(cmd)

            # 非阻塞循环
            while True:
                line = proc.stdout.readline()
                if line:
                    print(line, end="")
                    log_f.write(line)

                # 检查是否结束
                if proc.poll() is not None:
                    break

                # 检查 STOP
                if should_stop(command_file):
                    print("\n[Worker] STOP detected! Killing task...")
                    stop_proc(proc)
                    clear_stop(command_file)
                    break

                time.sleep(1)

            proc.wait()
            log_f.close()

            print(f"[Worker] Log saved to: {log_file}")

            if proc.returncode == 0:
                print("[Worker] Task finished successfully")
                mark_done(command_file, idx, lines)
            else:
                print(f"[Worker] Task failed with code {proc.returncode}")
                print("[Worker] Will retry later...")

            holder = start_holder()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")