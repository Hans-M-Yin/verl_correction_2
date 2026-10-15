#!/usr/bin/env python3
"""Keep a process alive with negligible CPU usage."""

import signal
import threading


stop_event = threading.Event()


def handle_stop(signum, frame):
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print("Process is alive. Press Ctrl+C or send SIGTERM to stop.", flush=True)
    stop_event.wait()
    print("Stopping.", flush=True)


if __name__ == "__main__":
    main()
