import unittest


def load_tests(loader, standard_tests, pattern):
    # 这层 tests/tests 是早期保留的重复测试目录。
    # 为了避免 unittest discover 把同名模块导入两次，这里显式返回空测试集。
    return unittest.TestSuite()
