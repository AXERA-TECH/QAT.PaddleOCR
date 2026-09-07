import sys

from pytorchocr.training.logging import training_log


def test_training_log_mirrors_stdout_and_stderr_and_appends(tmp_path, capsys):
    with training_log(tmp_path):
        print("first stdout")
        print("first stderr", file=sys.stderr)

    with training_log(tmp_path):
        print("second stdout")

    captured = capsys.readouterr()
    assert "first stdout" in captured.out
    assert "first stderr" in captured.err
    assert "second stdout" in captured.out
    assert (tmp_path / "train.log").read_text(encoding="utf-8") == (
        "first stdout\nfirst stderr\nsecond stdout\n"
    )


def test_training_log_restores_streams_after_exception(tmp_path, capsys):
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with training_log(tmp_path):
            print("before failure")
            raise RuntimeError("expected failure")
    except RuntimeError:
        pass

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    log = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert "before failure" in log
    assert "RuntimeError: expected failure" in log
