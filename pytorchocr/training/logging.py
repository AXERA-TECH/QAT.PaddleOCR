"""Logging helpers for command-line training entry points."""

from contextlib import contextmanager
from pathlib import Path
import sys
import traceback


class TeeStream:
    """Write a stream to both the original stream and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(
            stream.isatty()
            for stream in self._streams
            if hasattr(stream, "isatty")
        )

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


@contextmanager
def training_log(output_dir, filename="train.log"):
    """Mirror stdout and stderr to ``output_dir/filename``.

    The file is opened in append mode so a resumed or repeated run does not
    destroy the earlier terminal output. The original streams remain active,
    which preserves interactive progress and shell redirection behavior.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    try:
        yield log_path
    except BaseException:
        # The interpreter prints an unhandled exception only after leaving
        # this context, when stderr has already been restored. Preserve one
        # copy in the run log here and let the interpreter print the normal
        # terminal traceback after the exception is re-raised.
        traceback.print_exc(file=log_file)
        log_file.flush()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
