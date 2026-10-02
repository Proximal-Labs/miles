"""Resolve local files or modal://volume/path dataset sources."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


def _download(volume_name, remote_path, destination, environment):
    # Local datasets do not require the optional Modal SDK or credentials.
    import modal

    volume = modal.Volume.from_name(volume_name, environment_name=environment)
    with destination.open("wb") as stream:
        for chunk in volume.read_file(remote_path):
            stream.write(chunk)


@contextmanager
def dataset_paths(sources, *, environment="main"):
    """Fetch a fresh snapshot; temporary files live only while examples are loaded."""
    with TemporaryDirectory(prefix="reward-hacking-datasets-") as directory:
        paths = []
        for index, source in enumerate(sources):
            source = str(source)
            if not source.startswith("modal://"):
                paths.append(Path(source))
                continue
            uri = urlsplit(source)
            if not uri.netloc or uri.path in {"", "/"} or uri.query or uri.fragment:
                raise ValueError(f"Expected modal://volume/path/to/dataset.jsonl: {source}")
            path = Path(directory) / f"{index}.jsonl"
            _download(uri.netloc, uri.path, path, environment)
            paths.append(path)
        yield paths
