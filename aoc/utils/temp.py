import os
import pathlib
import tempfile
import time


class TempCache:
    def __init__(self, filename: str, mode="r", folder=None, retain_seconds=3600, verbose=False):
        if folder is None:
            folder = tempfile.gettempdir()
        self.folder = folder
        self.path = pathlib.Path(self.folder) / filename
        self.mode = mode
        self._file_object = None
        self.retain_seconds = retain_seconds
        self.verbose = verbose
    #

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def age(self):
        try:
            epoch_modified = os.path.getmtime(self.path)
            age = time.time() - epoch_modified
            return age
        except FileNotFoundError:
            return None

    def _remove_if_obsolete(self):
        age = self.age()
        if age is not None and age > self.retain_seconds:
            self.vprint(f"Removing file: {self.path}.")
            self.path.unlink()

    def __enter__(self):
        self._remove_if_obsolete()
        self._file_object = open(self.path, mode=self.mode)
        return self._file_object

    def __exit__(self, *args):
        self._file_object.close()


if __name__ == '__main__':
    fn = "hmm.txt"
    with TempCache(fn, "a", retain_seconds=0, verbose=True) as cache:
        cache.write("foo")
    with TempCache(fn, "a", retain_seconds=1, verbose=True) as cache:
        cache.write("\n")
        cache.write("bar")

    with TempCache(fn, "r", retain_seconds=1, verbose=True) as cache:
        print(cache.read().splitlines())
    #
