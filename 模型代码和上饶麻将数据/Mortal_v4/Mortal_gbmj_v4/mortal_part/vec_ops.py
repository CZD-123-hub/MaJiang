def vec_add_assign(lhs, rhs):
    """
    将两个包含相同元素的列表相加并在第一个列表中更新
    Python 中的列表是动态类型的，可能在运行时可能会遇到类型不兼容的问题

    :param lhs: 需要更新的列表
    :param rhs: 被加到lhs中的列表
    """
    # 确保两个列表长度相同
    if len(lhs) != len(rhs):
        raise ValueError("Lists must have the same length")

    # 使用 zip 函数将两个列表的对应元素配对，并更新 lhs 列表
    for i, (l, r) in enumerate(zip(lhs, rhs)):
        try:
            lhs[i] = l + r
        except TypeError:
            raise TypeError(f"Incompatible types: {type(l)} and {type(r)} at index {i}")


if __name__ == '__main__':
    # 示例使用
    lhs_list = [1, 2, 3]
    rhs_list = [4, 5, 6]
    vec_add_assign(lhs_list, rhs_list)
    print(lhs_list)  # 输出: [5, 7, 9]
