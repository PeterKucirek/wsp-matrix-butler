import numpy as np
import pandas as pd


def coerce_matrix(matrix, allow_raw=True, force_square=True):
    """
    Infers a NumPy array from given input

    Args:
        matrix:
        allow_raw (bool, optional): Defaults to ``True``.
        force_square (bool, optional): Defaults to ``True``.

    Returns:
        numpy.ndarray:
            A 2D ndarray of type float32
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
        return wide.values.astype(np.float32)

    if not allow_raw:
        raise NotImplementedError()

    matrix = np.array(matrix, dtype=np.float32)
    assert len(matrix.shape) == 2
    i, j = matrix.shape
    assert i == j

    return matrix


def expand_array(a, n, axis=None):
    if axis is None: new_shape = [dim + n for dim in a.shape]
    else:
        new_shape = []
        for i, dim in enumerate(a.shape):
            dim += n if i == axis else 0
            new_shape.append(dim)

    out = np.zeros(new_shape, dtype=a.dtype)

    indexer = [slice(0, dim) for dim in a.shape]
    out[indexer] = a

    return out


def to_fortran(matrix, file, n_columns=None, min_index=1, force_square=True):
    assert min_index >= 1
    array = coerce_matrix(matrix, force_square=force_square)

    if n_columns is not None and n_columns > array.shape[1]:
        extra_columns = n_columns - array.shape[1]
        array = expand_array(array, extra_columns, axis=1)

    with open(file, mode='wb') as writer:
        rows, columns = array.shape
        temp = np.zeros([rows, columns + 1], dtype=np.float32)
        temp[:, 1:] = array

        index = np.arange(min_index, rows + 1, dtype=np.int32)
        # Mask the integer binary representation as floating point
        index_as_float = np.frombuffer(index.tobytes(), dtype=np.float32)
        temp[:, 0] = index_as_float

        temp.tofile(writer)


def read_fortran_rectangle(file, n_columns, zones=None, tall=False, reindex_rows=False, fill_value=None):
    with open(file, mode='rb') as reader:
        n_columns = int(n_columns)

        matrix = np.fromfile(reader, dtype=np.float32)
        rows = len(matrix) // (n_columns + 1)
        assert len(matrix) == (rows * (n_columns + 1))

        matrix.shape = rows, n_columns + 1

        # Convert binary representation from float to int, then subtract 1 since FORTRAN uses 1-based positional
        # indexing
        row_index = np.frombuffer(matrix[:, 0].tobytes(), dtype=np.int32) - 1
        matrix = matrix[:, 1:]

        if zones is None:
            if tall:
                matrix.shape = matrix.shape[0] * matrix.shape[1]
            return matrix

        if isinstance(zones, (int, np.int_)):
            matrix = matrix[: zones, :zones]

            if tall:
                matrix.shape = zones * zones
            return matrix

        nzones = len(zones)
        matrix = matrix[: nzones, : nzones]
        row_labels = zones.take(row_index[:nzones])
        matrix = pd.DataFrame(matrix, index=row_labels, columns=zones)

        if reindex_rows:
            matrix = matrix.reindex_axis(zones, axis=0, fill_value=fill_value)

        if tall:
            return matrix.stack()
        return matrix
