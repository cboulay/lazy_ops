"""Provides a class to allow for lazy transposing and slicing operations on h5py datasets and zarr arrays

## Usage:

from lazy_ops import DatasetView

# h5py #
import h5py
dsetview = DatasetView(dataset) # dataset is an instance of h5py.Dataset
view1 = dsetview.lazy_slice[1:40:2,:,0:50:5].lazy_transpose([2,0,1]).lazy_slice[8,5:10]

# zarr #
import zarr
zarrview = DatasetView(zarray) # dataset is an instance of zarr.core.Array
view1 = zview.lazy_slice[1:10:2,:,5:10].lazy_transpose([0,2,1]).lazy_slice[0:3,1:4]

# reading from view on either h5py or zarr
A = view1[:]          # Brackets on DataSetView call the h5py or zarr slicing method, returning the data
B = view1.dsetread()  # same as view1[:]

# iterating on either h5yy or zarr
for ib in view.lazy_iter(axis=1):
    print(ib[0])

"""
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Union
import h5pickle as h5py


installed_dataset_types = h5py.Dataset


class DatasetView(metaclass=ABCMeta):
    """
    A DatasetView object comprises
    - a file-backed dataset (e.g., h5py Dataset),
    - an internal state to map the input original dataset to the output after applying lazy transformations
    - support for python slicing semantics (via __getitem__) which will return lazily the sliced, reshaped & transposed dataset.
    - member functions returning views of lazy transformations (e.g., transpose, reshape) of the original dataset
      The goal is to implement numpy ndarray shape manipulation methods reshape, transpose, swapaxes, squeeze:
      https://numpy.org/doc/stable/reference/arrays.ndarray.html#shape-manipulation

    The internal state variables are _dset_slices, _dset_ax_splits, and _axis_map
    _dset_slices:
        Slices to apply to the dataset when returning data.
        Each element can be a slice object, an integer iterator, or a single integer.
        len(dset_slices) <= len(dataset). If < len(dataset), then trailing dims are taken in full.
        The sliced dset ndim may be less than the original dataset if any of the slices
        are a single integer.

    _sliced_reshape:
        Splits (reshapes) individual axes in the sliced dataset when returning data.
        A dict, with keys being integer indices into the sliced dataset,
        and values being a tuple indicating the shape of the split. e.g., if axis 2
        has len=20 after slicing, then is reshaped to (5, 4), _dset_ax_splits = {2: (5, 4)}
        (assuming no other splits).

    _axis_map:
        Transposes the sliced & split dataset when returning data.
        A list of ints. Each integer maps the output axis to the axis in the sliced & split dataset.
        Not every axis in the sliced & split dataset must be mapped. For example, if slicing with an integer
        results in an axis with length=1, then that axis may be omitted from output. Conversely, if splitting
        (reshaping) results in an axis with length=1 then it is likely that axis should be retained in the output.

    TODO: Maybe someday _sliced_reshape and _axis_map can be reimplemented with something like
    numpy's strides, and this can be even more flexible.
    """
    # def __getnewargs_ex__(self):
    #     return (), {'dataset': self._dataset, 'dataset_slices': self._dset_slices,
    #                 'sliced_reshape': self._sliced_reshape, 'axis_map': self._axis_map}

    def __new__(cls,
                dataset: installed_dataset_types = None,
                dataset_slices=np.index_exp[:],
                sliced_reshape=None,
                axis_map=None):
        """
        Args:
          dataset: the underlying dataset
          dataset_slices: see class docstring
          sliced_reshape: see class docstring
          axis_map: see class docstring
        Returns:
          lazy object
        """
        if cls == DatasetView:
            if isinstance(dataset, h5py.Dataset):
                return DatasetViewh5py(dataset=dataset)
            elif HAVE_ZARR:
                if isinstance(dataset, zarr.core.Array):
                    return DatasetViewzarr(dataset=dataset)
            elif str(z1).find("zarr") != -1:  # TODO: What is z1?
                raise TypeError("To use DatasetView with a zarr array install zarr: \n pip install zarr\n")
            raise TypeError("DatasetView requires either an h5py dataset or a zarr array as first argument")
        else:
            return super().__new__(cls)

    def __init__(self,
                 dataset: installed_dataset_types = None,
                 dataset_slices=np.index_exp[:],
                 sliced_reshape=None,
                 axis_map=None):
        """
        Args:
          dataset: the underlying dataset
          sliced_reshape: See class docstring
          axis_map: See class docstring
        """
        self._lazy_slice_call = False  # If True then the next slice will return a view
        self._dataset = dataset
        self._dset_slices = self._sanitize_slices(dataset_slices)
        self._sliced_shape = None  # Shape of sliced dataset
        self._calculate_sliced_shape()
        self._sliced_reshape = sliced_reshape if sliced_reshape is not None else self._sliced_shape
        self._axis_map = list(axis_map) if axis_map is not None else list(range(len(self._sliced_reshape)))
        self._shape = tuple([self._sliced_reshape[_] for _ in self._axis_map])

    def __getstate__(self):
        """Save the current name and a reference to the root file object."""
        save_state = {
            'dataset': self._dataset,
            'dataset_slices': self._dset_slices,
            'sliced_reshape': self._sliced_reshape,
            'axis_map': self._axis_map
        }
        return save_state

    def __setstate__(self, state):
        """File is reopened by pickle. Create a dataset and steal its identity"""
        self.__init__(**state)

    def _sanitize_slices(self, slices_):
        slices_ = self._slice_tuple(slices_)                              # Ensure tuple
        slices_ = self._ellipsis_slices(slices_, ndim=self.dataset.ndim)  # Expand any ellipsis to full (list of) slices
        slices_ = self._fill_slices(slices_, with_shape=self.dataset.shape)  # Convert any slice(None) to integers.
        return slices_

    @classmethod
    def _slice_size(cls, sl):
        if isinstance(sl, slice):
            sl_start = sl.start if sl.start is not None else 0
            sl_step = sl.step if sl.step is not None else 1
            sl_size = 1 + (sl.stop - sl_start - 1) // sl_step if sl.stop != sl_start else 0
        elif isinstance(sl, int):
            sl_size = None  # Does not make it to the output
        else:  # iterator of integers
            sl_size = len(sl)
        return sl_size

    def _calculate_sliced_shape(self):
        sl_dset_shape = [self._slice_size(_) for _ in self._dset_slices]
        sl_dset_shape = [_ for _ in sl_dset_shape if _ is not None]  # Drop integer slices from shape
        # Add on trailing yet-to-be-sliced dimensions
        sl_dset_shape.extend(self.dataset.shape[len(self._dset_slices):])
        self._sliced_shape = tuple(sl_dset_shape)

    @property
    def lazy_slice(self):
        """ Indicator for lazy_slice calls """
        self._lazy_slice_call = True
        return self

    @property
    def dataset(self):
        return self._dataset

    @property
    def shape(self):
        if self._shape is None:
            self._calculate_sliced_shape()
            # TODO: Check self._sliced_reshape is not None
            self._shape = tuple([self._sliced_reshape[_] for _ in self._axis_map])
        return self._shape

    def __len__(self):
        return self.len()

    def len(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @classmethod
    def _slice_tuple(cls, slices_):
        """ Ensure that the passed-in object is a tuple
        Args:
          slices_: A slice object, an int, a list of ints, an ndarray, or a tuple
          (tuple contents not checked but each should be slice, int, or int iterator, or ndarray)
        Returns:
          The slice object tuple
        """
        if isinstance(slices_, (slice, int, np.ndarray)):
            slices_ = slices_,
        else:
            slices_ = *slices_,
        return slices_

    @classmethod
    def _slice2iter(cls, sl):
        if isinstance(sl, slice):
            sl = list(range(sl.start, sl.stop, sl.step))
        elif isinstance(sl, int):
            sl = [sl]
        # TODO: handle boolean indexing.
        return sl

    @classmethod
    def _mixed2flatgen(cls, mixed_list):
        for item in mixed_list:
            if isinstance(item, tuple):
                for subitem in item: yield subitem
            else:
                yield item

    def _ellipsis_slices(self, slices_, ndim=None):
        """ Change Ellipsis dimensions to slices
        Args:
          slices_: The new slice (tuple)
          ndim: Number of dimensions. default: Uses self.dataset.ndim
        Returns:
          equivalent slices with Ellipsis expanded
        """
        ellipsis_count = sum(s==Ellipsis for s in slices_ if not isinstance(s, np.ndarray))
        if ellipsis_count == 1:
            ellipsis_index = slices_.index(Ellipsis)
            if ellipsis_index == len(slices_) - 1:
                slices_ = slices_[:-1]
            else:
                ndim = self.dataset.ndim if ndim is None else self.ndim
                n_legit_slices = len([_ for _ in slices_ if _ not in [Ellipsis, None]])
                num_ellipsis_dims = ndim - n_legit_slices
                slices_ = slices_[:ellipsis_index] + np.index_exp[:] * num_ellipsis_dims + slices_[ellipsis_index + 1:]
        elif ellipsis_count > 0:
            raise IndexError("Only a single Ellipsis is allowed")
        return slices_

    def _fill_slices(self, slices_, with_shape=None):
        """
        :param slices_: iterable of slice objects and/or integers
        :param with_shape: shape of object being sliced. default: use self.dataset.shape
        :return: Same as input except slice objects of form slice(None) have been replaced with
                 slice objects defined by the correct integers.
        """
        slices_ = list(slices_)
        with_shape = with_shape if with_shape is not None else self.dataset.shape
        slices_ += [slice(None)] * (len(with_shape) - len([_ for _ in slices_ if _ is not None]))
        shape_idx = 0  # Index into with_shape. Do not increment for slice == None.
        for ax, sl in enumerate(slices_):
            if isinstance(sl, slice):
                if shape_idx < len(with_shape):
                    slices_[ax] = slice(*sl.indices(with_shape[shape_idx]))
                else:
                    # Not enough info in with_shape.
                    # TODO: Error as more not-None slices were provided than there are items in shape
                    slices_[ax] = slices_[ax]
            if sl is not None:
                shape_idx += 1
        return tuple(slices_)

    def __getitem__(self, new_slices):
        """ Slice DatasetView
        Args:
          new_slice: the new slice to compose with the lazy instance's previous slice and transformation operations
        Returns:
          lazy obj if this is the first slice since obj.lazy_slice() or all items in new_slice are Ellipsis or None,
          otherwise return the sliced dataset (i.e., a numpy array, including reading uncached data from disk).
        Notes:
          Fancy indexing not supported. multi-dimensional numpy arrays will be converted to flat lists.
        """
        new_slices = self._slice_tuple(new_slices)
        new_slices = tuple([_ if not issubclass(type(_), np.ndarray) else _.flatten().tolist() for _ in new_slices])
        # If there is one or more None entries, and all other slice items are innocuous
        #  (i.e., Ellipsis, (slice(None) if self._lazy_slice_call)
        #  then the slice operation is simply a dim expansion and we can shortcut to self.reshape
        ignore_slices = (Ellipsis, slice(None)) if self._lazy_slice_call else (Ellipsis,)
        shortcut = None in new_slices and all([_ in (ignore_slices + (None,)) for _ in new_slices])
        if shortcut:
            n_inserted = 0
            new_shape = list(self.shape)
            for sl_ix, sl in enumerate(new_slices):
                if sl is None:
                    new_shape = new_shape[:sl_ix + n_inserted] + (1,) + new_shape[sl_ix + n_inserted:]
                    n_inserted += 1
            return self.reshape(new_shape)

        # This is not a simple dim expansion, proceed with slicing
        new_slices, new_sliced_reshape, new_axis_map = self._compose_slices(new_slices)
        new_obj = DatasetView(self.dataset, dataset_slices=new_slices,
                              sliced_reshape=new_sliced_reshape, axis_map=new_axis_map)
        if self._lazy_slice_call:
            self._lazy_slice_call = False
            return new_obj
        return new_obj.dsetread()

    def lazy_iter(self, axis=0):
        """ lazy iterator over the first axis
            Modifications to the items are not stored
        """
        for ix in range(self._shape[axis]):
            yield self.lazy_slice[(*np.index_exp[:]*axis, ix)]

    def __call__(self, new_slice):
        """  allows lazy_slice function calls with slice objects as input"""
        return self.__getitem__(new_slice)

    @classmethod
    def _compose_slice_pair(cls, old_sl, new_sl, old_len=None):
        if old_len is None:
            old_len = cls._slice_size(old_sl)
            if old_len is None:  # integer
                old_len = 1

        if isinstance(new_sl, slice):
            if isinstance(old_sl, slice):
                upd_sl = slice(
                    min(old_sl.start + old_sl.step * new_sl.start, old_sl.stop),
                    min(old_sl.start + old_sl.step * new_sl.stop, old_sl.stop),
                    new_sl.step * old_sl.step)
            else:
                # old_sl is an iterator of integers, so we simply select from those integers.
                upd_sl = old_sl[new_sl]
        elif isinstance(new_sl, int):
            if new_sl >= old_len or new_sl <= ~old_len:
                raise IndexError("Index %d out of range, dim %d of size %d"
                                 % (new_sl, ax_ix, old_len))
            if isinstance(old_sl, slice):
                upd_sl = old_sl.start + old_sl.step * (new_sl % old_len)
            else:
                # old_sl is an iterator of integers, so we just choose the one of those we want.
                upd_sl = old_sl[new_sl]
        elif new_sl is None:
            print("This shouldn't be reached. Please notify devs if you see this.")
        else:
            # Handle new_sl being an iterator of integers.
            if not all(isinstance(el, int) for el in new_sl):
                # Convert boolean array to iterator of integers.
                if new_sl.dtype.kind != 'b':
                    raise ValueError("Indices must be either integers or booleans")
                else:
                    # boolean indexing
                    if len(new_sl) != old_len:
                        raise IndexError("Length of boolean index $d must be equal to size %d"
                                         % (len(new_sl), old_len))
                    upd_sl = new_sl.nonzero()[0]
            if any(el >= old_len or el <= ~old_len for el in new_sl):
                raise IndexError("Index %s out of range, of size %d"
                                 % (str(new_sl), old_len))
            if isinstance(old_sl, slice):
                upd_sl = tuple(old_sl.start + old_sl.step * (ind % old_len)
                               for ind in new_sl)
            else:
                # old_sl is an iterator of integers
                upd_sl = tuple(old_sl[ind] for ind in new_sl)
            # If upd_sl is a long integer iterator with even step sizes then compress it to a slice object
            step_sizes = np.diff(upd_sl)
            if len(step_sizes) > 2 and len(np.unique(step_sizes)) == 1:
                upd_sl = slice(upd_sl[0], upd_sl[-1] + 1, step_sizes[0])

        return upd_sl

    def _compose_slices(self, new_slices):
        """  Composes new_slice(s) with previous slices, reshapes, and transposes.
        Args:
          new_slices: The new slice. Can be a single object or a tuple of objects.
            Each object must be either a slice object, a single integer, an iterator of integers, a boolean array
            or None to indicate a new axis.
        Returns:
           new_slices, new_reshapes, new_axis_map
        Notes:
          This returns a new slice tuple, new reshapes list, and new axis_map list that can be used to initialize
          a new DatasetView object with the same dataset, and have the result reflect the composition of previous
          transformations and the new slice. It does not modify self in place.
        """
        new_slices = self._slice_tuple(new_slices)  # Ensure tuple
        new_slices = tuple([_ if not issubclass(type(_), np.ndarray) else _.flatten().tolist() for _ in new_slices])
        new_slices = self._ellipsis_slices(new_slices, ndim=self.ndim)  # Expand any ellipsis to full (list of) slices
        b_noop = [_ == slice(None) for _ in new_slices]
        if all(b_noop):
            # If all slices are no-ops (i.e., they return all samples) then we can shortcut.
            return self._dset_slices, self._sliced_reshape, self._axis_map

        new_slices = self._fill_slices(new_slices, with_shape=self.shape)

        # Keep only slices without None. We will deal with None for dim expansion later.
        true_slices = [_ for _ in new_slices if _ is not None]

        # inverse transpose - gives slices on the reshaped dataset
        rshp_slices = [sl for _, sl in sorted(zip(self._axis_map, true_slices))]

        # inverse reshape:
        # This is tricky for axes that have been split or merged.
        # Instead of inverse reshape now, simply identify splits/merges then handle case-by-case during compose.
        reshape_targets, reshape_ok = self._map_shapes(self._sliced_reshape, old_shape=self._sliced_shape)
        assert reshape_ok, "It shouldn't be possible to trigger this."

        # Compose slices - in reverse order
        # Any old slices that are integers will eliminate the corresponding axes before reaching reshape level.
        # So we only need to compose with old slices that are not int, but keep track of where they came from.
        dset_sl_idx = [ix for ix, _ in enumerate(self._dset_slices) if not isinstance(_, int)]
        new_dset_slices = [self._dset_slices[_] for _ in dset_sl_idx]
        new_reshape = list(self._sliced_reshape)  # New reshape on dataset[new_dset_slices]. None indicates int slice
        sl_ax = len(rshp_slices) - 1
        while sl_ax >= 0:
            targ = reshape_targets[sl_ax]
            if isinstance(targ, list):
                # For axes that are merged...
                # Take original slices, convert to int iterators, use np.ravel_multi_index and flatten
                merge_int_iters = []
                merge_dset_shapes = []
                for t_ax in targ:
                    merge_int_iters.append(self._slice2iter(new_dset_slices[t_ax]))
                    merge_dset_shapes.append(self.dataset.shape[t_ax])
                # merge_int_iters may need to be reshaped to make them broadcastable with each other
                if len(set([len(_) for _ in merge_int_iters])) > 1:
                    merge_axes = range(len(merge_int_iters))
                    for _ix, int_iter in enumerate(merge_int_iters):
                        merge_int_iters[_ix] = np.expand_dims(int_iter, tuple(np.setdiff1d(merge_axes, _ix)))
                else:
                    print("TODO: previous slices were already integer iterators resulting from slice on merged axes.")
                    # Anything to do or will it just work?
                old_merge_sl = np.ravel_multi_index(merge_int_iters, merge_dset_shapes).flatten()
                new_merge_sl = self._compose_slice_pair(old_merge_sl, rshp_slices[sl_ax])
                new_ds_sl = np.unravel_index(new_merge_sl, merge_dset_shapes)
                for t_ax, _sl in zip(targ, new_ds_sl):
                    new_dset_slices[t_ax] = _sl.tolist()
                new_reshape[sl_ax] = len(new_merge_sl)
                if isinstance(rshp_slices[sl_ax], int):
                    new_reshape[sl_ax] = None
            else:
                old_sl = new_dset_slices[targ]
                b_targ = [_ == targ for _ in reshape_targets]
                if sum(b_targ) > 1:
                    # For axes that have been split
                    rshp_idx = np.where(b_targ)[0]
                    old_sl_as_iter = self._slice2iter(old_sl)
                    old_iter_ = np.reshape(old_sl_as_iter, self._sliced_reshape[b_targ])
                    new_iter_ = old_iter_[tuple(_ for ix, _ in enumerate(rshp_slices) if b_targ[ix])]
                    new_dset_slices[targ] = new_iter_.flatten().tolist()
                    new_iter_ix = 0  # Index into new_iter_.shape, needed so we can skip int-indexed dimensions.
                    for r_ix in rshp_idx:
                        if isinstance(rshp_slices[r_ix], int):
                            new_reshape[r_ix] = None
                        else:
                            new_reshape[r_ix] = new_iter_.shape[new_iter_ix]
                            new_iter_ix += 1
                    sl_ax = rshp_idx[0]  # Choose firstr rshp ax so sl_ax decrement will go to new slice
                else:
                    # Normal composition
                    new_dset_slices[targ] = self._compose_slice_pair(old_sl, rshp_slices[sl_ax])
                    new_reshape[sl_ax] = self._slice_size(new_dset_slices[targ])  # <- None if slice is int

            sl_ax -= 1

        # Update new_reshape and create new_axis_map
        # - expand axes where sl == None
        # - drop axes where sl is int
        new_axis_map = list(self._axis_map)
        # We previously excluded the None slices when calculating slice compositions:
        # new_slices --drop None--> true_slices --self._axis_map--> rshp_slices <--> new_reshape
        # Find the None entries in new_slices, map them to new_reshape, and insert <1> in new_reshape
        map_ix = 0
        expanded_axes = []
        for ax, sl in enumerate(new_slices):
            if sl is not None:
                map_ix += 1
            else:
                targ = self._axis_map[map_ix]
                expanded_axes.append(targ)  # Will be inserted into new_shape later.
                # Increment items in new_axis_map that are >= expanded axis
                new_axis_map = [_ if _ < targ else _ + 1 for _ in new_axis_map]
                # Insert an entry to the new axis in the axis map
                new_axis_map.insert(ax, targ)

        # Insert expanded_axes into new_reshape. Expanded axes need to be sorted and reversed prior to insertion.
        for ex_ax in reversed(sorted(expanded_axes)):
            new_reshape.insert(ex_ax, 1)

        # Drop None entries -- these are from new integer-slices on old axes
        drop_targs = [ax for ax, _ in enumerate(new_reshape) if _ is None]  # targs to drop from new_axis_map
        new_reshape = [_ for _ in new_reshape if _ is not None]
        for dt in reversed(sorted(drop_targs)):
            new_axis_map.remove(dt)
            new_axis_map = [_ if _ < dt else _ - 1 for _ in new_axis_map]

        return new_dset_slices, new_reshape, new_axis_map

    def dsetread(self):
        """ Returns the data
        Returns:
          numpy array
        """
        result = self.dataset[self._dset_slices]
        # Reshape if necessary
        if not np.array_equal(self._sliced_reshape, result.shape):
            result = result.reshape(self._sliced_reshape)
        # Transpose if necessary
        if not np.array_equal(self._axis_map, range(len(result.shape))):
            result = result.transpose(self._axis_map)
        return result

    @property
    def T(self):
        """ Same as lazy_transpose() """
        return self.lazy_transpose()

    def lazy_transpose(self, axis_order=None):
        """ Array lazy transposition, no axis_order reverses the order of dimensions
        Args:
          axis_order: permutation order for transpose
        Returns:
          lazy object
        """

        if axis_order is None:
            axis_order = tuple(reversed(range(len(self._axis_map))))

        return DatasetView(self.dataset, self._dset_slices, self._sliced_reshape, axis_order)

    def transpose(self, axis_order=None):
        return self.lazy_transpose(axis_order=axis_order)

    def _map_shapes(self, new_shape, old_shape=None):
        """
        Map how new_shape is derived from old_shape.
        return is a tuple. The first item, reshape_targets, is a list where each element identifies
        which element(s) in old_shape it is derived. Elements in old_shape can be repeated in cases
        where the old axis was split, or elements in new_shape can be a tuple that point to multiple
        elements in old_shape to indicate where the old axes were merged.
        :param new_shape: list of integers
        :param old_shape: list of integers
        :return: reshape_targets, map_successful
        """
        old_shape = np.array(old_shape) if old_shape is not None else np.copy(self.shape)
        reshape_ok = True
        new_ix = len(new_shape) - 1
        old_ix = len(old_shape) - 1
        reshape_targets = [None] * len(new_shape)
        while new_ix >= 0:
            if old_ix < 0:
                assert new_shape[new_ix] == 1, "Leading 'extra' dims should always be length 1"
                reshape_targets[new_ix] = 0
                new_ix -= 1
            elif new_shape[new_ix] == old_shape[old_ix]:
                reshape_targets[new_ix] = old_ix
                new_ix -= 1
                old_ix -= 1
            elif new_shape[new_ix] < old_shape[old_ix]:
                # split old axis: Calculate how many axes old_shape[old_ix] is split to in new_shape
                n_split = np.where(np.cumprod(new_shape[new_ix::-1]) == old_shape[old_ix])[0]
                if len(n_split) == 0:
                    # Current element in new_shape could not be combined with leading elements to produce a length
                    # equal to the current element in old_shape. This happens when attempting to split n > 1 axes
                    reshape_ok = False
                    break
                new_ix0 = new_ix - n_split[-1]
                reshape_targets[new_ix0:new_ix + 1] = [old_ix] * (new_ix - new_ix0 + 1)
                old_ix -= 1
                new_ix = new_ix0 - 1
            else:  # new_shape[new_ix] > old_shape[old_ix]
                # merge old axis
                # Calculate how many axes of old_shape[:old_ix] are being merged
                old_offset = np.where(np.cumprod(old_shape[old_ix::-1]) == new_shape[new_ix])[0]
                if len(old_offset) == 0:
                    # Current element in old_shape could not be combined with preceding elements to produce a length
                    # equal to the current element in new_shape. This happens when attempting to merge into n > 1 axes
                    reshape_ok = False
                    break
                reshape_targets[new_ix] = list(range(old_ix - old_offset[0], old_ix + 1))
                old_ix = old_ix - old_offset[0] - 1
                new_ix -= 1

        return reshape_targets, reshape_ok

    def reshape(self, new_shape):
        """
        :param new_shape:
        :return: A new DatasetView object with the view of the dataset having the new shape.
        Notes:
        
        """
        # https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/shape.c#L364

        if np.array_equal(new_shape, self.shape):
            return DatasetView(self.dataset, self._dset_slices, self._sliced_reshape, self._axis_map)

        new_shape = np.array(new_shape)
        if -1 in new_shape:
            known_size = np.abs(np.prod(new_shape))
            assert (self.size % known_size) == 0
            unknown_ix = np.where(new_shape == -1)[0]
            assert len(unknown_ix) == 1, "reshape only supports one unknown dimension length (-1)."
            new_shape[unknown_ix[0]] = self.size // known_size

        assert np.prod(new_shape) == np.prod(self.shape), "reshape cannot change the number of elements."

        # Identify which entries in new_shape correspond to which entries in self.shape.
        reshape_targets, reshape_ok = self._map_shapes(new_shape)

        if not reshape_ok:
            # TODO: logger.warning("reshape only supports even many-to-one merges or one-to-many splits. Try chaining multiple reshape operations.")
            return self.dsetread().reshape(new_shape)

        # Map reshape operations back through _axis_map.
        unmap_reshape_targets = []
        for targ in reshape_targets:
            if isinstance(targ, list):
                merge_list = [self._axis_map[_] for _ in targ]
                # Can only merge ascending adjacent axes
                reshape_ok &= np.all(np.diff(merge_list) == 1)
                unmap_reshape_targets.append(merge_list[0])
            else:
                unmap_reshape_targets.append(self._axis_map[targ])

        if not reshape_ok:
            # TODO: logger.warning("Requested reshape operation cannot be composed with previous state. Returning ndarray")
            return self.dsetread().reshape(new_shape)

        # Compose mapped reshape operations and old reshape operations. Easy to do given limitations above.
        new_sliced_reshape = new_shape[np.argsort(unmap_reshape_targets)]

        # Update previous transpose operations to reflect new reshaping.
        new_map = np.copy(unmap_reshape_targets)
        prev_targ_ax = min(new_map)
        while prev_targ_ax <= max(new_map):
            match_inds = np.where(new_map == prev_targ_ax)[0]
            n_matches = len(match_inds)
            new_map[new_map > prev_targ_ax] = new_map[new_map > prev_targ_ax] + n_matches - 1
            new_map[match_inds] = prev_targ_ax + range(n_matches)
            prev_targ_ax += len(match_inds)

        return DatasetView(self.dataset, dataset_slices=self._dset_slices,
                           sliced_reshape=new_sliced_reshape,
                           axis_map=new_map)

    def __array__(self):
        """ Convert to numpy array
        """
        return np.atleast_1d(self.dsetread())

    def _slice_shape(self, slice_):
        """  For an slice returned by _slice_composition function, finds the shape
        Args:
          slice_: The slice and int_index object
        Returns:
          slice_shape: Shape of the dataset after slicing with slice_key
          slice_key: An equivalent slice tuple with positive starts and stops
          int_index: a nested tuple, int_index records the information needed by dsetread to access data
                                     Each element of int_index, denoted ind is given by:
                                     ind[2] is the dataset axis at which the integer index operates
                                     ind[1] is the value of the integer index entered by the user
                                     ind[0] is the lazy_axis at which the integer index operates
                                                  ,the lazy_axis is the axis number had the operations
                                                  been carried out by h5py instead of lazy_ops
          axis_order: removes the elements of current axis_order where integer indexing has been applied
        """
        int_ind = slice_[1]
        slice_ = self._slice_tuple(slice_[0])
        # convert the slice to regular slices that only contain integers (or None)
        slice_ = [slice(*sl.indices(self.dataset.shape[self.axis_order[ix]])) if isinstance(sl, slice) else sl
                  for ix, sl in enumerate(slice_)]

        slice_shape = ()
        int_index = ()
        axis_order = ()
        for ix, sl in enumerate(slice_):
            if isinstance(sl, slice):
                if sl.step < 1:
                    raise ValueError("Slice step parameter must be positive")
                if sl.stop < sl.start:
                    slice_[ix] = slice(sl.stop, sl.stop, sl.step)
                    sl = slice_[ix]
                slice_shape += (1 + (sl.stop - sl.start - 1) // sl.step if sl.stop != sl.start else 0,)
                axis_order += (self.axis_order[ix],)
            elif isinstance(sl, int):
                int_index += ((i, sl, self.axis_order[ix]),)
            else:
                # slice_[i] is an iterator of integers
                slice_shape += (len(sl),)
                axis_order += (self.axis_order[ix],)

        # Drop integer indices
        slice_ = tuple(sl for sl in slice_ if not isinstance(sl, int))
        axis_order += tuple(self.axis_order[len(axis_order) + len(int_index)::])
        int_index += int_ind
        slice_shape += self.dataset.shape[len(slice_shape) + len(int_index)::]

        return slice_shape, slice_, int_index, axis_order


def lazy_transpose(dset: installed_dataset_types, axes=None):
    """ Array lazy transposition, not passing axis argument reverses the order of dimensions
    Args:
      dset: h5py dataset
      axes: permutation order for transpose
    Returns:
      lazy transposed DatasetView object
    """
    if axes is None:
        axes = tuple(reversed(range(len(dset.shape))))

    return DatasetView(dset).lazy_transpose(axis_order=axes)


class DatasetViewh5py(DatasetView, h5py.Dataset):

    def __new__(cls, dataset):

        _self = super().__new__(cls)
        h5py.Dataset.__init__(_self, dataset.id)
        return _self

    def __getnewargs__(self):
        return (self._dataset,)

    # @property
    # def file_info(self):
        # # Shouldn't the mixin take care of this?
        # return self.dataset.file_info


try:
    import zarr
    from .lazy_loading_zarr import DatasetViewzarr
    installed_dataset_types = Union[installed_dataset_types, zarr.core.Array]
    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False
    DatasetViewzarr = None