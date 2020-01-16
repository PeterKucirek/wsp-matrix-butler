from __future__ import division
import os
from os import path
import sqlite3 as sqlite
from datetime import datetime as dt
from warnings import warn
from contextlib import contextmanager
from pkg_resources import parse_version

import numpy as np
import pandas as pd

from .api import ButlerOverwriteWarning, MatrixEntry

LEGACY_PANDAS = parse_version(pd.__version__) < parse_version('0.24')


class MatrixButler(object):

    _MATRIX_EXTENSION = '.bin'
    _SUBDIRECTORY_NAME = 'emmebin'
    _DB_NAME = "matrix_directory.sqlite"

    def __init__(self, *args):
        butler_path, db, zone_system, fortran_max_zones = args
        self._path = butler_path
        self._connection = db
        self._zone_system = zone_system
        self._max_zones_fortran = fortran_max_zones

        self._committing = True

    @staticmethod
    def create(parent_directory, zone_system, fortran_max_zones):
        """Creates a new (or clears and initializes and existing) MatrixButler.

        Args:
            parent_directory (unicode): The parent directory in which to keep the Butler.
            zone_system (pandas.Int64Index or List[int]): The zone system to conform to.
            fortran_max_zones (int): The total number of zones expected by the FORTRAN matrix reader.

        Returns:
            MatrixButler instance.
        """
        zone_system = pd.Int64Index(zone_system)
        fortran_max_zones = int(fortran_max_zones)

        butler_path = path.join(parent_directory, MatrixButler._SUBDIRECTORY_NAME)

        if not path.exists(butler_path):
            os.makedirs(butler_path)

        dbfile = path.join(butler_path, MatrixButler._DB_NAME)
        db_exists = path.exists(dbfile)  # Connecting to a non-existent file will create the file, so cache this first
        db = sqlite.connect(dbfile)
        db.row_factory = sqlite.Row

        if db_exists:
            fortran_max_zones_existing, zone_system_existing = MatrixButler._preload(db)
            existing_is_compatible = fortran_max_zones == fortran_max_zones_existing and zone_system.equals(
                zone_system_existing)
            if not existing_is_compatible:
                msg = 'Existing matrix cache not compatible with current zone system and will be cleared of any ' \
                      'stored matrix files. Cache directory is `%s`' % parent_directory
                warn(ButlerOverwriteWarning(msg))

                for fn in os.listdir(butler_path):
                    if fn.endswith(MatrixButler._MATRIX_EXTENSION):
                        fp = path.join(butler_path, fn)
                        os.remove(fp)
                MatrixButler._clear_tables(db)
                MatrixButler._create_tables(db, zone_system, fortran_max_zones)
            # When the db exits AND is compatible, there's no need to create the tables
        else:
            MatrixButler._create_tables(db, zone_system, fortran_max_zones)

        return MatrixButler(butler_path, db, zone_system, fortran_max_zones)

    @staticmethod
    def _preload(db):
        sql = """
        SELECT *
        FROM properties
        WHERE name="max_zones_fortran"
        """
        result = list(db.execute(sql))
        fortran_max_zones = int(result[0]['value'])

        sql = """
        SELECT *
        FROM zone_system
        """
        result = list(db.execute(sql))
        zone_system = pd.Int64Index([int(record['zone']) for record in result])

        return fortran_max_zones, zone_system

    @staticmethod
    def _clear_tables(db):
        db.execute("DROP TABLE IF EXISTS properties;")
        db.execute("DROP TABLE IF EXISTS zone_system;")
        db.execute("DROP TABLE IF EXISTS matrices;")
        db.commit()

    @staticmethod
    def _create_tables(db, zone_system, fortran_max_zones):
        sql = """
        CREATE TABLE properties(
        name VARCHAR NOT NULL PRIMARY KEY,
        value VARCHAR
        );
        """
        db.execute(sql)

        sql = """
        INSERT INTO properties
        VALUES (?, ?)
        """
        db.execute(sql, ('max_zones_fortran', fortran_max_zones))
        db.execute(sql, ('zone_partition', -1))

        sql = """
        CREATE TABLE zone_system(
        number INT NOT NULL PRIMARY KEY,
        zone INT
        );
        """
        db.execute(sql)

        sql = """
        INSERT INTO zone_system
        VALUES (?, ?)
        """
        for i, zone in enumerate(zone_system):
            zone = int(zone)  # Cast to Python INT, because SQLite doesn't like NumPy integers.
            db.execute(sql, (i, zone))

        sql = """
        CREATE TABLE matrices(
        id VARCHAR NOT NULL PRIMARY KEY,
        description VARCHAR,
        timestamp VARCHAR,
        type VARCHAR
        );
        """
        db.execute(sql)

        sql = """
        CREATE TABLE matrix_numbers(
        id VARCHAR NOT NULL,
        slice INT NOT NULL,
        number INT,
        PRIMARY KEY (id, slice)
        );
        """
        db.execute(sql)

        db.commit()

    @staticmethod
    def connect(parent_directory):
        """Connect to an existing MatrixButler, without initializing it

        Args:
            parent_directory (unicode): The parent directory in which to find the MatrixButler.

        Returns:
            MatrixButler instance.

        Raises:
            IOError: if a MatrixButler cannot be found at the given parent directory.
        """
        butler_path = path.join(parent_directory, MatrixButler._SUBDIRECTORY_NAME)
        if not os.path.exists(butler_path):
            raise IOError('No matrix butler found at `%s`' % parent_directory)

        dbfile = path.join(butler_path, MatrixButler._DB_NAME)
        if not os.path.exists(dbfile):
            raise IOError('No matrix butler found at `%s`' % parent_directory)

        db = sqlite.connect(dbfile)
        db.row_factory = sqlite.Row
        fortran_max_zones, zone_system = MatrixButler._preload(db)

        return MatrixButler(butler_path, db, zone_system, fortran_max_zones)

    def __iter__(self):
        sql = """
        SELECT *
        FROM matrices
        """
        result = list(self._connection.execute(sql))
        for item in result:
            yield MatrixEntry(item['id'], item['description'], item['timestamp'], item['type'])

    def __contains__(self, item):
        sql = """
        SELECT id
        FROM matrices
        WHERE id=?
        """
        result = list(self._connection.execute(sql, [item]))
        return len(result) > 0

    def __len__(self):
        sql = """
        SELECT *
        FROM matrices
        """
        return len(list(self._connection.execute(sql)))

    def __getitem__(self, item):
        sql = """
        SELECT id
        FROM matrices
        WHERE id=?
        """
        result = list(self._connection.execute(sql, [item]))
        if len(result) < 1:
            raise KeyError(item)

        item = result[0]
        return MatrixEntry(item['id'], item['description'], item['timestamp'], item['type'])

    def to_frame(self):
        """Returns a representation of the butler's contents as a pandas DataFrame"""
        uids, descriptions, timestamps, types = [], [], [], []
        for entry in self:
            uids.append(entry.uid)
            descriptions.append(entry.description)
            timestamps.append(entry.timestamp)
            types.append(entry.type_name)
        df = pd.DataFrame(index=uids)
        df.index.name = 'uid'
        df['description'] = descriptions
        df['timestamp'] = timestamps
        df['type_name'] = types

        return df

    @property
    def zone_system(self):
        return self._zone_system[...]  # Ellipses to make a shallow copy

    @property
    def zone_partition(self):
        sql = """
        SELECT *
        FROM properties
        WHERE name="zone_partition"
        """
        result = list(self._connection.execute(sql, ''))
        zone_partition = int(result[0]['value'])
        return None if zone_partition <= 0 else zone_partition

    @zone_partition.setter
    def zone_partition(self, val):
        if val is not None:
            val = int(val)
            assert val in self._zone_system
        else:
            val = -1
        sql = """
        INSERT OR REPLACE INTO properties
        VALUES (?, ?)
        """
        self._connection.execute(sql, ('zone_partition', val))
        if self._committing:
            self._connection.commit()

    def _matrix_file(self, n):
        return path.join(self._path, "mf%s%s" % (n, MatrixButler._MATRIX_EXTENSION))

    @contextmanager
    def batch_operations(self):
        """Context-manager for writing several matrices in one batch. Reduces write time per matrix by committing 
        changes to the DB at the end of the batch write. The time savings can be quite significant, as the DB-write is 
        normally 50% of the time per matrix write.

        Yields: 
            None
        """
        # This snippet is just in case this function is called within its own context (e.g. someone turns on
        # batch mode while it's already on).
        if not self._committing:
            yield
            return

        self._committing = False

        try:
            yield
        finally:
            self._connection.commit()
            self._committing = True

    def lookup_numbers(self, unique_id, squeeze=True):
        """Looks up file number(s) (e.g. 50 corresponds to "mf50.bin") of a given matrix.

        Args:
            unique_id (str): The ID of the matrix to look up
            squeeze (bool): If True, and only one matrix number corresponds to the unique ID, then the result will
                be a single integer. Otherwise, a list of integers is returned.

        Returns: 
            Int or List[int], depending on results and `squeeze`
        """

        sql = """
        SELECT *
        FROM matrix_numbers
        WHERE id=?
        """
        result = list(self._connection.execute(sql, [unique_id]))
        if not result:
            raise KeyError(unique_id)

        numbers = [int(item['number']) for item in result]

        if len(numbers) == 1 and squeeze:
            numbers = numbers[0]
        return numbers

    def is_sliced(self, unique_id):
        """Checks if a matrix is sliced on-disk or not.

        Args:
            unique_id (str): The ID of the matrix to checl

        Returns: 
            bool: True if matrix is sliced, False otherwise.

        Raises: 
            KeyError: if unique_id is not in the butler.
        """
        return len(self.lookup_numbers(unique_id, squeeze=False)) > 1

    def _next_numbers(self, n):
        sql = "SELECT number FROM matrix_numbers;"
        existing_numbers = {int(item['number']) for item in list(self._connection.execute(sql))}

        numbers = []
        cursor = 1
        for _ in range(n):
            while cursor in existing_numbers:
                cursor += 1
            numbers.append(cursor)
            cursor += 1

        return numbers

    def _check_lookup(self, unique_id, n, partition):
        if partition:
            n += 1

        try:
            numbers = self.lookup_numbers(unique_id, squeeze=False)
            if len(numbers) != n:
                # Overwriting an existing matrix with different number of slices causes that matrix to be completely
                # replaced (deleted and then overwritten)
                self.delete_matrix(unique_id)
                numbers = self._next_numbers(n)
        except KeyError:
            numbers = self._next_numbers(n)

        return numbers

    def _write_matrix_record(self, unique_id, numbers, description, type_name):
        timestamp = dt.now()
        sql = """
        INSERT OR REPLACE INTO matrices
        VALUES (?, ?, ?, ?)
        """
        self._connection.execute(sql, (unique_id, description, timestamp, type_name))

        for slice_, number in enumerate(numbers):
            sql = """
            INSERT OR REPLACE INTO matrix_numbers
            VALUES (?, ?, ?)
            """
            self._connection.execute(sql, (unique_id, slice_, number))

        if self._committing:
            self._connection.commit()

    def _validate_slice_args(self, n_slices, partition):
        n_slices = int(n_slices)
        assert n_slices >= 1

        if partition and self.zone_partition is None:
            warn("Cannot partition a matrix when MatrixButler.zone_partition is None.")
            return n_slices, False

        return n_slices, partition

    # def _expand_matrix(self, matrix, n_slices, partition):
    #     if n_slices == 1 and not partition:
    #         rows, cols = matrix.shape
    #         assert rows == cols
    #         padding = self._max_zones_fortran - rows
    #         if padding > 0:
    #             return expand_array(matrix, padding, axis=None)
    #         return matrix
    #     else:
    #         cols = matrix.shape[1]
    #         padding = self._max_zones_fortran - cols
    #         if padding > 0:
    #             return expand_array(matrix, padding, axis=1)
    #     return matrix

    def _write_matrix_files(self, matrix_array, files, partition):

        # matrix = self._expand_matrix(matrix, len(files), partition)

        remainder, remainder_file = None, None
        if partition:
            slice_end = self._zone_system.get_loc(self.zone_partition) + 1

            remainder = matrix_array[slice_end:, :]
            remainder_file = files.pop()  # Remove from the list of files

            matrix_array = matrix_array[:slice_end, :]

        n_slices = len(files)
        slices = np.array_split(matrix_array, n_slices, axis=0) if n_slices > 1 else [matrix_array]

        min_index = 1
        for slice_, file_ in zip(slices, files):
            self._to_binary_file(slice_, file_, min_index)
            min_index += slice_.shape[0]
        if partition:
            self._to_binary_file(remainder, remainder_file, min_index)

    def init_matrix(self, unique_id, description="", type_name="", fill=True, n_slices=1, partition=False):
        """Registers a new (or zeros an old) matrix with the butler.

        Args:
            unique_id (str): The unique identifier for this matrix.
            description (str):  A brief description of the matrix.
            type_name (str): Type categorization of the matrx.
            fill (bool): If False, empty (0-byte) files will be initialized. Otherwise, 0-matrix files will be created.
            n_slices (int): Number of slices (on-disk) for multi-processing.
            partition (bool): Flag whether to partition the matrix before saving, based on self.zone_partition
        """
        n_slices, partition = self._validate_slice_args(n_slices, partition)

        numbers = self._check_lookup(unique_id, n_slices, partition)
        files = [self._matrix_file(n) for n in numbers]

        if fill:
            shape = [len(self._zone_system)] * 2
            matrix = np.zeros(shape, dtype=np.float32)
            self._write_matrix_files(matrix, files, partition)

        self._write_matrix_record(unique_id, numbers, description, type_name)

    def load_matrix(self, unique_id, tall=False):
        """Gets a matrix from the butler, optionally saving into an Emmebank.

        Args:
            unique_id (str): The name you gave to the butler for safekeeping.
            tall (bool):

        Returns: 
            DataFrame or None, depending on whether `target_mfid` is given.

        Raises:
            KeyError: if unique_id is not in the butler.
        """

        is_sliced = self.is_sliced(unique_id)
        numbers = self.lookup_numbers(unique_id, squeeze=False)

        matrices = []
        for number in numbers:
            fp = self._matrix_file(number)
            subframe = self._from_binary_file(fp)
            matrices.append(subframe)
        matrix = pd.concat(matrices, axis=0) if is_sliced else matrices[0]

        if not matrix.index.equals(self._zone_system):
            matrix = matrix.reindex(self._zone_system, fill_value=0.0)

        if tall:
            return matrix.stack()
        return matrix

    def save_matrix(self, dataframe_or_array, unique_id, description="", type_name="", n_slices=1, partition=False,
                    reindex=True, fill_value=0.0):
        """Passes a matrix to the butler for safekeeping.

        Args:
            dataframe_or_array (DataFrame or str): Specifies the matrix to save. If basestring, it is assumed to
                refer to a matrix in an Emmebank (see `emmebank`). Otherwise, a square DataFrame is required.
            unique_id (str): The unique identifier for this matrix.
            description (str): A brief description of the matrix.
            type_name (str): The string type
            n_slices (int): Number of slices (on-disk) for multi-processing.
            partition (bool): Flag whether to partition the matrix before saving, based on self.zone_partition
            reindex (bool): Flag to indicate if partial matrices are accepted when supplying a DataFrame. If False,
                AssertionError will be raised when one of the DataFrame's axes doesn't match the Butler's zone system.
            fill_value (float): The fill value to be used when reindexing (see 'reindex' flag) or filling Emme infinity
                (see `fill_eminf` flag)
        """

        n_slices, partition = self._validate_slice_args(n_slices, partition)

        if isinstance(dataframe_or_array, pd.DataFrame):
            if not dataframe_or_array.index.equals(self._zone_system):
                if not reindex:
                    raise AssertionError()
                dataframe_or_array = dataframe_or_array.reindex_axis(self._zone_system, fill_value=fill_value, axis=0)
            if not dataframe_or_array.columns.equals(self._zone_system):
                if not reindex:
                    raise AssertionError()
                dataframe_or_array = dataframe_or_array.reindex_axis(self._zone_system, fill_value=fill_value, axis=1)
        else:
            raise TypeError()
        matrix = coerce_matrix(dataframe_or_array, allow_raw=True, force_square=True)
        assert matrix.shape == (len(self._zone_system),) * 2

        numbers = self._check_lookup(unique_id, n_slices, partition)
        files = [self._matrix_file(n) for n in numbers]
        self._write_matrix_files(matrix, files, partition)

        self._write_matrix_record(unique_id, numbers, description, type_name)

    def query_type(self, type_name):
        """Gets a list of matrix IDs by their type name

        Args:
            type_name (str): The type name to query

        Returns:
            A list of matching IDs
        """
        sql = """
        SELECT id FROM matrices
        WHERE type=?;
        """
        return [item['id'] for item in self._connection.execute(sql, [type_name])]

    def matrix_metadata(self, unique_id):
        """Looks up a single matrix record and returns its metadata (description, type, timestamp) in a dictionary

        Args:
            unique_id (str): The unique ID of the matrix to lookup

        Returns: 
            Dict: corresponding to the matrix record. Keys are 'id', 'description', 'timestamp', and 'type'

        Raises: 
            KeyError: if unique ID not in the butler.
        """

        sql = """
        SELECT id, description, type, timestamp FROM matrices
        WHERE id=?;
        """
        results = list(self._connection.execute(sql, [unique_id]))
        if not results:
            raise KeyError(unique_id)
        row = results[0]
        return dict(row)

    def delete_matrix(self, unique_id):
        """Deletes a matrix from the butler's directory

        Args:
            unique_id (str): The unique identifier of the matrix to delete.

        Raises:
            KeyError: if unique_id cannot be found.
        """

        numbers = self.lookup_numbers(unique_id, squeeze=False)
        for n in numbers:
            fp = self._matrix_file(n)
            os.remove(fp)

        sql = """
        DELETE FROM matrices
        WHERE id=?;
        """
        self._connection.execute(sql, [unique_id])

        sql = """
        DELETE FROM matrix_numbers
        WHERE id=?;
        """
        self._connection.execute(sql, [unique_id])

        if self._committing:
            self._connection.commit()

    def __del__(self):
        self._connection.commit()
        self._connection.close()

    def slice_matrix(self, unique_id, n_slices=1, partition=None):
        """
        Slices a matrix into chunks along its rows (on-disk) for use in mutli-processing.

        Does nothing if matrix is already sliced.

        Args:
            unique_id (str): The ID of the matrix to slice
            n_slices (int): The number of slices to make
            partition (bool): Flag whether to partition the matrix before saving, based on self.zone_partition
        """
        prior_n_slices = len(self.lookup_numbers(unique_id, squeeze=False))
        expected_slices = n_slices + 1 if partition else n_slices
        if prior_n_slices == expected_slices: return  # Do nothing if already sliced to the called number

        n_slices, partition = self._validate_slice_args(n_slices, partition)

        matrix = self.load_matrix(unique_id)
        metadata = self.matrix_metadata(unique_id)

        with self.batch_operations():
            self.delete_matrix(unique_id)
            self.save_matrix(matrix, unique_id, metadata['description'], metadata['type'], n_slices=n_slices,
                             partition=partition)

    def unslice_matrix(self, unique_id):
        """
        "Un-slices" (i.e. concatenates) a sliced matrix in the butler. Does nothing if the matrix is not sliced.

        Args:
            unique_id (str): The ID of the matrix to slice.

        """
        if not self.is_sliced(unique_id): return  # Do nothing is matrix is already not sliced
        matrix = self.load_matrix(unique_id)
        metadata = self.matrix_metadata(unique_id)

        with self.batch_operations():
            self.delete_matrix(unique_id)
            self.save_matrix(matrix, unique_id, metadata['description'], metadata['type'], n_slices=1)

    # region matrices

    def _to_binary_file(self, array, file_, min_index=1):
        assert min_index >= 1

        rows, columns = array.shape

        # Mask the integer binary representation as floating point
        index = np.arange(min_index, min_index + rows, dtype=np.int32)
        index_as_float = np.frombuffer(index.tobytes(), dtype=np.float32).reshape(rows, 1)

        to_concat = [index_as_float, array]
        extra_columns = self._max_zones_fortran - columns
        if extra_columns > 0:
            padding = np.zeros(shape=(rows, extra_columns), dtype=np.float32)
            to_concat.append(padding)

        expanded = np.concatenate(to_concat, axis=1)

        with open(file_, mode='wb') as writer:
            expanded.tofile(writer)

    def _from_binary_file(self, file_):
        with open(file_, mode='rb') as reader:
            raw_floats = np.fromfile(reader, dtype=np.float32)

        rows = len(raw_floats) // (self._max_zones_fortran + 1)
        assert len(raw_floats) == (rows * (self._max_zones_fortran + 1))

        raw_floats.shape = rows, self._max_zones_fortran + 1
        row_offsets = np.frombuffer(raw_floats[:, 0].tobytes(), dtype=np.int32) - 1
        row_index = self.zone_system.take(row_offsets[: len(self.zone_system)])  # MLOGIT likes to pad out to max zones, which we don't need.

        real_columns = len(self.zone_system)
        matrix = raw_floats[: len(self.zone_system):, 1: real_columns + 1].copy()  # Drop extra columns and make a deep copy to force GC to cleanup the raw

        frame = pd.DataFrame(matrix, index=row_index, columns=self.zone_system)
        return frame


def coerce_matrix(matrix, allow_raw=True, force_square=True):
    """Infers a NumPy array from given input

    Args:
        matrix:
        allow_raw (bool, optional): Defaults to ``True``.
        force_square (bool, optional): Defaults to ``True``.

    Returns:
        numpy.ndarray: A 2D ndarray of type float32
    """
    if isinstance(matrix, pd.DataFrame):
        if force_square:
            assert matrix.index.equals(matrix.columns)
        return matrix.values.astype(np.float32)
    elif isinstance(matrix, pd.Series):
        assert matrix.index.nlevels == 2, "Cannot infer a matrix from a Series with more or fewer than 2 levels"
        wide = matrix.unstack()

        union = wide.index | wide.columns
        wide = wide.reindex_axis(union, fill_value=0.0, axis=0).reindex_axis(union, fill_value=0.0, axis=1)
        if LEGACY_PANDAS:
            return wide.values.astype(np.float32)
        else:
            return wide.to_numpy(copy=True).astype(np.float32)

    if not allow_raw:
        raise NotImplementedError()

    matrix = np.array(matrix, dtype=np.float32)
    assert len(matrix.shape) == 2
    i, j = matrix.shape
    assert i == j

    return matrix
    # endregion
