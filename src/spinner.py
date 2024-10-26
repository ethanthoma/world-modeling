import itertools
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional

RATE: float = 0.080
FRAMES: List[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class SharedState:
    def __init__(self, text: str = ""):
        self.current_text = text


def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def spinner_task(
    shared_state: SharedState,
    lock: threading.Lock,
    stop_event: threading.Event,
):
    spinner_chars = itertools.cycle(FRAMES)
    while not stop_event.is_set():
        with lock:
            sys.stdout.write("\r")
            sys.stdout.write(f"{next(spinner_chars)} {shared_state.current_text}")
            sys.stdout.flush()
        time.sleep(RATE)


class SpinnerWriter:
    def __init__(
        self,
        shared_state: SharedState,
        lock: threading.Lock,
        stop_event: threading.Event,
    ):
        self.shared_state = shared_state
        self.lock = lock
        self.stop_event = stop_event

    def write(self, text: str) -> None:
        with self.lock:
            self.shared_state.current_text = text

    def stop(self) -> None:
        self.stop_event.set()


@contextmanager
def spinner():
    if not sys.stdout.isatty():
        sys.stdout.write("\r")
        yield None
        return

    shared_state = SharedState()
    stop_event = threading.Event()
    lock = threading.Lock()

    hide_cursor()

    thread = threading.Thread(
        target=spinner_task, args=(shared_state, lock, stop_event)
    )
    thread.daemon = True
    thread.start()

    writer = SpinnerWriter(shared_state, lock, stop_event)

    try:
        yield writer
    finally:
        stop_event.set()
        with lock:
            sys.stdout.write("\r")
            sys.stdout.write(" " * (len(shared_state.current_text) + 2))
            sys.stdout.write("\r")
            sys.stdout.flush()
        thread.join(timeout=1.0)
        show_cursor()