import zarr
from .lazy_loading import DatasetView


class DatasetViewzarr(DatasetView):

    # def __new__(cls, dataset):
    #     _self = super().__new__(cls)
    #     zarr.core.Array.__init__(_self, dataset.store, path=dataset.path)
    #     return _self
    #
    # def __getnewargs__(self):
    #     return (self._dataset,)

    def __init__(self, dataset: zarr.core.Array = None, **kwargs):
        DatasetView.__init__(self, dataset=dataset, **kwargs)

    def get_writable_copy(self):
        return self

    def make_copyable(self):
        pass

    def _refresh_filename(self):
        pass

    def move_or_copy_file(self, new_path, mode='r'):
        pass

    @property
    def using_tempfile(self):
        return False

