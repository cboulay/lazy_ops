import zarr
from .lazy_loading import DatasetView

class DatasetViewzarr(DatasetView, zarr.core.Array):

    def __new__(cls, dataset):
        _self = super().__new__(cls)
        zarr.core.Array.__init__(_self, dataset.store, path=dataset.path)
        return _self

    def __getnewargs__(self):
        return (self._dataset,)

    def get_writable_copy(self):
        return self
