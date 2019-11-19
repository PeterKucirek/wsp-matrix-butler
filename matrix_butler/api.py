class MatrixEntry(object):

    __slots__ = ['uid', 'description', 'timestamp', 'type_name']

    def __init__(self, *args):
        self.uid, self.description, self.timestamp, self.type_name = args

    def __repr__(self):
        return "MatrixEntry('%s')" % self.uid

    def __str__(self):
        return self.uid


class ButlerOverwriteWarning(RuntimeWarning):
    pass

