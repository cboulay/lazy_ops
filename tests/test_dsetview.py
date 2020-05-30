import numpy as np
from lazy_ops import DatasetView, lazy_transpose
import secrets
from numpy.testing import assert_array_equal
import unittest
import tempfile
from functools import wraps
import h5py
import h5pickle
import zarr


# Define decorator to iterate over dset_list
def dset_iterator(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        for self.dset, self.dsetview in zip(self.dset_list, self.dsetview_list):
            f(self, *args, **kwargs)
    return wrapper


class LazyOpsBase(object):

    srand = secrets.SystemRandom()

    @classmethod
    def _slices(cls, shape):
        """ find an appropriate tuple of slices """
        return tuple(slice(cls.srand.randint(~s-1, s+1), cls.srand.randint(~s-1, s+1),
                           cls.srand.randint(1, s)) for s in shape)

    @classmethod
    def _axis(cls, n):
        """ find an appropriate tuple of axes given number of dimensions """
        axes = list(range(n))
        cls.srand.shuffle(axes)
        return axes

    @classmethod
    def _int_indexing(cls, shape):
        """ find an appropriate tuple of integers """
        return tuple(cls.srand.randint(0, s-1) for s in shape)

    @classmethod
    def _array_indexing(cls, shape):
        """ find an appropriate tuple with a single array index """
        single_array_dim = cls.srand.randrange(0, len(shape))
        single_array_len = cls.srand.randrange(0, shape[single_array_dim])
        single_array_indexing = sorted(cls.srand.sample(range(shape[single_array_dim]),
                                                        single_array_len))
        return tuple(slice(None, None, None) if i != single_array_dim else single_array_indexing
                     for i in range(len(shape)))

    @classmethod
    def _bool_indexing(cls, shape):
        """ find an appropriate tuple with a single array index """
        single_array_dim = cls.srand.randrange(0, len(shape))
        single_bool_indexing = np.array(cls.srand.choices([True,False], k=shape[single_array_dim]))
        return tuple(slice(None, None, None) if i != single_array_dim else single_bool_indexing
                     for i in range(len(shape)))

    @classmethod
    def _slices_and_int(cls, shape):
        """ find an appropriate tuple of slices and integers """
        return tuple(slice(cls.srand.randint(~s-1, s+1), cls.srand.randint(~s-1, s+1),
                           cls.srand.randint(1, s))
                     if cls.srand.choice([True, False]) else
                     cls.srand.randint(0, s-1)
                     for s in shape)

    @classmethod
    def _slices_and_array(cls, shape, single_array_dim):
        """ find an appropriate tuple of slices and a single array index"""
        single_array_len = cls.srand.randrange(0, shape[single_array_dim])
        single_array_indexing = sorted(cls.srand.sample(range(shape[single_array_dim]),
                                                        single_array_len))
        return tuple(slice(cls.srand.randint(~s-1, s+1), cls.srand.randint(~s-1, s+1),
                           cls.srand.randint(1, s))
                     if i != single_array_dim else
                     single_array_indexing
                     for i, s in enumerate(shape))

    @classmethod
    def _slices_and_bool(cls, shape, single_array_dim):
        """ find an appropriate tuple of slices and a single array index"""
        single_bool_indexing = np.array(cls.srand.choices([True, False], k=shape[single_array_dim]))
        return tuple(slice(cls.srand.randint(~s-1, s+1), cls.srand.randint(~s-1, s+1),
                           cls.srand.randint(1, s))
                     if i != single_array_dim else
                     single_bool_indexing
                     for i, s in enumerate(shape))


    ##########################################
    #  basic tests                           #
    ##########################################

    @dset_iterator
    def test_dsetview_array(self):
        # test __array__ read
        assert_array_equal(self.dset, self.dsetview)

    @dset_iterator
    def test_dsetview_nonlazy_slicing(self):
        # test __getitem__ read
        assert_array_equal(self.dset, self.dsetview[()])
        # test __getitem__ single slice read
        assert_array_equal(self.dset, self.dsetview[:])
        slices = self._slices(self.dset.shape)
        assert_array_equal(self.dset[slices], self.dsetview[slices])

    ##########################################
    #  tests for single lazy operation calls #
    ##########################################

    @dset_iterator
    def test_dsetview_lazy_slice(self):
        # test __getitem__ read after lazy_slice
        assert_array_equal(self.dset, self.dsetview.lazy_slice[()])
        # test __getitem__ read after lazy_slice, single slice
        assert_array_equal(self.dset, self.dsetview.lazy_slice[:])
        slices = self._slices(self.dset.shape)
        assert_array_equal(self.dset[slices], self.dsetview.lazy_slice[slices])

    @dset_iterator
    def test_dsetview_lazy_slice_lower_dimensions(self):
        for num_slice_dims in range(1, len(self.dset.shape)+1):
            slices = self._slices(self.dset.shape[:num_slice_dims])
            # test __getitem__ read specifying lower dimensions
            assert_array_equal(self.dset[slices], self.dsetview[slices])
            # test __getitem__ read after lazy_slice, single slice read for lower dimensions
            assert_array_equal(self.dset[slices], self.dsetview.lazy_slice[slices])

    @dset_iterator
    def test_dsetview_lazy_slice_int_indexing(self):
        for num_slice_dims in range(1, len(self.dset.shape)+1):
            indexing = self._int_indexing(self.dset.shape[:num_slice_dims])
            # test __getitem__ read specifying lower dimensions
            assert_array_equal(self.dset[indexing], self.dsetview[indexing])
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # int indexing only
            assert_array_equal(self.dset[indexing], self.dsetview.lazy_slice[indexing])

    @dset_iterator
    def test_dsetview_lazy_iter(self):
        for axis in range(len(self.dset.shape)):
            for i, dsetview_lazy_i in enumerate(self.dsetview.lazy_iter(axis=axis)):
                assert_array_equal(self.dset[(*np.index_exp[:]*axis, i)], dsetview_lazy_i)

    @dset_iterator
    def test_dsetview_lazy_transpose(self):
        axis = self._axis(len(self.dset.shape))
        # test DatasetView.lazy_transpose
        assert_array_equal(self.dset[()].transpose(axis), self.dsetview.lazy_transpose(axis))
        # test lazy_ops.lazy_transpose
        assert_array_equal(np.transpose(self.dset[()], axis),lazy_transpose(self.dsetview, axis))

    ###########################################
    # tests for multiple lazy slice calls     #
    ###########################################

    # multi lazy_slice using only slices
    @dset_iterator
    def test_dsetview_multi_lazy_slice(self):
        self._dsetview_multi_lazy_slice(self.dset, self.dsetview)

    @classmethod
    def _dsetview_multi_lazy_slice(cls, dset, dsetview):
        for num_slice_dims in range(1, len(dset.shape)+1):
            slices = cls._slices(dset.shape[:num_slice_dims])
            dset_new = dset[slices]
            dsetview_new = dsetview.lazy_slice[slices]
            # test __getitem__ read after lazy_slice for lower dimensions
            assert_array_equal(dset_new, dsetview_new)
            if np.prod(dset_new.shape) != 0:
                cls._dsetview_multi_lazy_slice(dset_new, dsetview_new)

    # multi lazy_slice using slices and int indexing
    @dset_iterator
    def test_dsetview_multi_lazy_slice_with_slice_and_int_indexing(self):
        self._dsetview_multi_lazy_slice_with_slice_and_int_indexing(self.dset, self.dsetview)

    @classmethod
    def _dsetview_multi_lazy_slice_with_slice_and_int_indexing(cls, dset, dsetview):
        for num_slice_dims in range(1, len(dset.shape)+1):
            indexing = cls._slices_and_int(dset.shape[:num_slice_dims])
            dset_new = dset[indexing]
            dsetview_new = dsetview.lazy_slice[indexing]
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # combination of slice and int indexing
            assert_array_equal(dset_new, dsetview_new)
            if np.prod(dset_new.shape) != 0:
                cls._dsetview_multi_lazy_slice_with_slice_and_int_indexing(dset_new, dsetview_new)

    ###########################################
    # tests for multiple lazy operation calls #
    ###########################################

    # lazy_slice with slices followed by lazy_transpose call
    @dset_iterator
    def test_dsetview_lazy_slice_lazy_transpose(self):
        for num_slice_dims in range(1, len(self.dset.shape)+1):
            slices = self._slices(self.dset.shape[:num_slice_dims])
            # test __getitem__ read after lazy_slice, single slice read for lower dimensions
            # followed by lazy_transpose
            axis = self._axis(len(self.dset[slices].shape))
            assert_array_equal(self.dset[slices].transpose(axis), self.dsetview.lazy_slice[slices].lazy_transpose(axis))

    # multi lazy_transpose calls
    @dset_iterator
    def test_dsetview_multi_lazy_transpose(self):
        remaining_transpose_calls = 10
        self._dsetview_multi_lazy_transpose(self.dset, self.dsetview, remaining_transpose_calls)

    @classmethod
    def _dsetview_multi_lazy_transpose(self, dset, dsetview, remaining_transpose_calls):
        axis = self._axis(len(dset.shape))
        dset_new = dset[()].transpose(axis)
        dsetview_new = dsetview.lazy_transpose(axis)
        # test DatasetView.lazy_transpose
        assert_array_equal(dset_new,dsetview_new)
        if remaining_transpose_calls > 0:
            self._dsetview_multi_lazy_transpose(dset_new, dsetview_new, remaining_transpose_calls - 1)

    # multi lazy_transpose and lazy_slice using only slices
    @dset_iterator
    def test_dsetview_multi_lazy_ops_with_slice_indexing(self):
        remaining_transpose_calls = 10
        self._dsetview_multi_lazy_ops_with_slice_indexing(self.dset, self.dsetview, remaining_transpose_calls)

    @classmethod
    def _dsetview_multi_lazy_ops_with_slice_indexing(cls, dset, dsetview, remaining_transpose_calls):
        for num_slice_dims in range(1, len(dset.shape) + 1):
            slices = cls._slices(dset.shape[:num_slice_dims])
            dset_new = dset[slices]
            dsetview_new = dsetview.lazy_slice[slices]
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            assert_array_equal(dset_new, dsetview_new)
            if np.prod(dset_new.shape) != 0:
                cls._dsetview_multi_lazy_ops_with_slice_indexing(dset_new, dsetview_new, remaining_transpose_calls)
        axis = cls._axis(len(dset.shape))
        dset_new = dset[()].transpose(axis)
        dsetview_new = dsetview.lazy_transpose(axis)
        # test DatasetView.lazy_transpose
        assert_array_equal(dset_new, dsetview_new)
        if remaining_transpose_calls > 0:
            cls._dsetview_multi_lazy_ops_with_slice_indexing(dset_new, dsetview_new, remaining_transpose_calls - 1)

    # multi lazy_transpose and lazy_slice using slices and int
    @dset_iterator
    def test_dsetview_multi_lazy_ops_with_slice_and_int_indexing(self):
        self._dsetview_multi_lazy_ops_with_slice_and_int_indexing(self.dset, self.dsetview)

    @classmethod
    def _dsetview_multi_lazy_ops_with_slice_and_int_indexing(cls, dset, dsetview):
        for num_slice_dims in range(1, len(dset.shape) + 1):
            slices = cls._slices_and_int(dset.shape[:num_slice_dims])
            dset_new = dset[slices]
            dsetview_new = dsetview.lazy_slice[slices]
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # combination of slice and int indexing
            assert_array_equal(dset_new, dsetview_new)
            if np.prod(dset_new.shape) != 0:
                cls._dsetview_multi_lazy_ops_with_slice_and_int_indexing(dset_new, dsetview_new)


class LazyOpsBaseh5py(object):

    ###########################################################
    #  tests for single lazy operation calls specific to h5py #
    ###########################################################

    @dset_iterator
    def test_dsetview_lazy_slice_bool(self):
        # test __getitem__ read after lazy_slice, single slice
        indexing = self._bool_indexing(self.dset.shape)
        assert_array_equal(self.dset[indexing], self.dsetview.lazy_slice[indexing])

    @dset_iterator
    def test_dsetview_lazy_slice_array_indexing(self):
        for num_slice_dims in range(1, len(self.dset.shape)+1):
            indexing = self._array_indexing(self.dset.shape[:num_slice_dims])
            # test __getitem__ read specifying lower dimensions
            assert_array_equal(self.dset[indexing], self.dsetview[indexing])
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # array indexing only
            assert_array_equal(self.dset[indexing], self.dsetview.lazy_slice[indexing])

    @dset_iterator
    def test_dsetview_lazy_slice_bool_indexing(self):
        for num_slice_dims in range(2, len(self.dset.shape)+1):
            # num_slice_dims starts from 2, dset[(1-D bool np.ndarray,)] is invalid in h5py
            # dset[(1-D bool np.ndarray, slice(None))] is valid
            indexing = self._bool_indexing(self.dset.shape[:num_slice_dims])
            # test __getitem__ read specifying lower dimensions
            assert_array_equal(self.dset[indexing], self.dsetview[indexing])
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # bool indexing only
            assert_array_equal(self.dset[indexing], self.dsetview.lazy_slice[indexing])

    ########################################################
    # tests for multiple lazy slice calls specific to h5py #
    ########################################################

    # multi lazy_slice using slices and array indexing
    @dset_iterator
    def test_dsetview_multi_lazy_slice_with_slice_and_array_indexing(self):
        remaining_slice_calls = 10
        array_dim = self.srand.randint(0, len(self.dset.shape)-1)
        self._dsetview_multi_lazy_slice_with_slice_and_array_indexing(self.dset, self.dsetview, remaining_slice_calls, array_dim)

    @classmethod
    def _dsetview_multi_lazy_slice_with_slice_and_array_indexing(cls, dset, dsetview, remaining_slice_calls, array_dim):
        for num_slice_dims in range(array_dim+1, len(dset.shape)+1):
            indexing = cls._slices_and_array(dset.shape[:num_slice_dims], array_dim)
            dset_new = dset[indexing]
            dsetview_new = dsetview.lazy_slice[indexing]
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # combination of slice and array indexing
            assert_array_equal(dset_new, dsetview_new)
            if np.prod(dset_new.shape) != 0 and remaining_slice_calls > 0:
                cls._dsetview_multi_lazy_slice_with_slice_and_array_indexing(dset_new, dsetview_new, remaining_slice_calls - 1, array_dim)

    # multi lazy_slice using slices and boolean array indexing
    @dset_iterator
    def test_dsetview_multi_lazy_slice_with_slice_and_bool_indexing(self):
        remaining_slice_calls = 4
        array_dim = self.srand.randint(1, len(self.dset.shape)-1)
        # array_dim starts from 1, for array_dim=0, dset[(1-D bool np.ndarray,)] is invalid in h5py
        # dset[(slice(None),1-D bool np.ndarray)] is valid
        self._dsetview_multi_lazy_slice_with_slice_and_bool_indexing(self.dset, self.dsetview, remaining_slice_calls, array_dim)

    @classmethod
    def _dsetview_multi_lazy_slice_with_slice_and_bool_indexing(cls, dset, dsetview, remaining_slice_calls, array_dim):
        for num_slice_dims in range(array_dim+1, len(dset.shape)+1):
            indexing = cls._slices_and_bool(dset.shape[:num_slice_dims], array_dim)
            dset_new = dset[indexing]
            dsetview_new = dsetview.lazy_slice[indexing]
            # test __getitem__ read after lazy_slice
            # for lower and all dimensions
            # combination of slice and bool indexing
            assert_array_equal(dset_new, dsetview_new)
            if np.prod(dset_new.shape) != 0 and remaining_slice_calls > 0:
                cls._dsetview_multi_lazy_slice_with_slice_and_bool_indexing(dset_new, dsetview_new, remaining_slice_calls - 1, array_dim)


class LazyOpsh5pickleTest(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        self.temp_file.close()
        h5py_file = h5py.File(self.temp_file.name, 'w')
        ndims = 7
        srand = secrets.SystemRandom()
        data = np.random.rand(*srand.choices(range(1, 90 // ndims), k=ndims))
        h5py_file.create_dataset(name='dset', data=data)
        h5py_file.close()

        self.h5pkl_file = h5pickle.File(self.temp_file.name, 'r')
        self.dset = self.h5pkl_file['/dset']
        self.dsetview = DatasetView(self.dset)

    def tearDown(self):
        self.temp_file.delete = True
        self.temp_file.close()

    def test_pkl_dset(self):
        """ Assert that pickling dset works."""
        import pickle
        dset_copy = pickle.loads(pickle.dumps(self.dset))
        assert_array_equal(self.dset, dset_copy)

    def test_pkl_dsetview(self):
        import pickle
        dsetview_copy = pickle.loads(pickle.dumps(self.dsetview))
        assert_array_equal(self.dsetview, dsetview_copy)
        assert_array_equal(self.dsetview.lazy_slice[..., 0], dsetview_copy[..., 0])


class LazyOpszarrTest(LazyOpsBase, unittest.TestCase):
    """ Class zarr array equality test """

    def setUp(self):
        ndims = 7
        num_datasets = 3

        self.temp_dir_zarr = tempfile.TemporaryDirectory(suffix=".zgroup")
        self.zarr_group = zarr.group(store=self.temp_dir_zarr.name, overwrite=True)
        self.dset_list = list(self.zarr_group.create_dataset(name='zarray'+str(i),
                              data=np.random.rand(*self.srand.choices(range(1, 90//ndims), k=ndims)))
                              for i in range(num_datasets))
        self.dsetview_list = list(DatasetView(self.dset_list[i]) for i in range(num_datasets))
        print(LazyOpszarrTest)

    def tearDown(self):
        self.temp_dir_zarr.cleanup()


class LazyOpsh5pyTest(LazyOpsBase, LazyOpsBaseh5py, unittest.TestCase):
    """ Class h5py dataset array equality test """

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        self.temp_file.close()
        self.h5py_file = h5py.File(self.temp_file.name, 'w')

        ndims = 7
        num_datasets = 3
        self.dset_list = list(self.h5py_file.create_dataset(name='dset' + str(i),
                              data=np.random.rand(*self.srand.choices(range(1, 90 // ndims), k=ndims)))
                              for i in range(num_datasets))
        self.dsetview_list = list(DatasetView(self.dset_list[i]) for i in range(num_datasets))

    def tearDown(self):
        self.temp_file.delete = True
        self.temp_file.close()
