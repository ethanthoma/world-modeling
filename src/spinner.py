import contextlib
import itertools
import sys
import threading
import time
from typing import Iterator, Optional

RATE: float = 0.080
FRAMES: list[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Shared_State:
    def __init__(self):
        self.current_text = ""


def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def spinner_task(
    shared_state: Shared_State,
    lock: threading.Lock,
    stop_event: threading.Event,
    output_stream,
):
    spinner_chars = itertools.cycle(FRAMES)
    while not stop_event.is_set():
        with lock:
            output_stream.write("\r\033[K")
            output_stream.write(f"{next(spinner_chars)} {shared_state.current_text}")
            output_stream.flush()
        time.sleep(RATE)


class Spinner_Stdout_Wrapper:
    def __init__(self, shared_state, lock):
        self.shared_state = shared_state
        self.lock = lock
        self.buffer = ""

    def write(self, text):
        with self.lock:
            self.buffer += text
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                for line in reversed(lines):
                    if line.strip():
                        self.shared_state.current_text = line
                        break
                if not lines[-1].endswith("\n"):
                    self.buffer = lines[-1]
                else:
                    self.buffer = ""
            else:
                self.shared_state.current_text = self.buffer

    def flush(self):
        pass

    def isatty(self):
        return True


@contextlib.contextmanager
def spinner():
    if not sys.stdout.isatty():
        sys.stdout.write("\r")
        yield
        return

    shared_state = Shared_State()
    stop_event = threading.Event()
    lock = threading.Lock()

    hide_cursor()

    original_stdout = sys.stdout
    sys.stdout = Spinner_Stdout_Wrapper(shared_state, lock)

    thread = threading.Thread(
        target=spinner_task, args=(shared_state, lock, stop_event, original_stdout)
    )
    thread.daemon = True
    thread.start()

    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        with lock:
            sys.stdout = original_stdout
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        show_cursor()
