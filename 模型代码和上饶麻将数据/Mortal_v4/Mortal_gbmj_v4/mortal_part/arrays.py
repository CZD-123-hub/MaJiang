import numpy as np


class Simple2DArray:
    def __init__(self, rows: int, cols: int, dtype=np.float32):
        """
        初始化二维数组，（维度743*34）
        :param rows: 行数
        :param cols: 列数
        :param dtype: 数据类型，默认为float
        """
        self.rows = rows
        self.cols = cols
        self.arr = np.zeros((rows, cols), dtype=dtype)

    def get(self, row: int, col: int) -> np.float32:
        """
        获取指定位置的值
        :param row: 行索引
        :param col: 列索引
        :return: 指定位置的值
        """
        return np.float32(self.arr[row, col])

    def rows(self) -> int:
        """
        获取行数
        :return: 数组的行数
        """
        return self.arr.shape[0]

    def fill(self, row: int, value: np.float32):
        """
        填充指定行
        :param row: 行索引
        :param value: 填充的值
        """
        self.fill_rows(row, 1, value)

    def fill_rows(self, row: int, n_rows: int, value: np.float32):
        """
        填充多行
        :param row: 起始行索引
        :param n_rows: 填充的行数
        :param value: 填充的值
        """
        self.arr[row:row + n_rows, :] = value

    def assign(self, row: int, col: int, value: np.float32):
        """
        在指定位置赋值
        :param row: 行索引
        :param col: 列索引
        :param value: 赋的值
        """
        self.arr[row, col] = value

    def assign_rows(self, row: int, col: int, n_rows: int, value: np.float32):
        """
        在指定列的多行赋值
        :param row: 起始行索引
        :param col: 列索引
        :param n_rows: 赋值的行数
        :param value: 赋的值
        """
        self.arr[row:row + n_rows, col] = value

    def build(self) -> np.ndarray:
        """
        返回numpy数组
        :return: numpy二维数组
        """
        return self.arr
