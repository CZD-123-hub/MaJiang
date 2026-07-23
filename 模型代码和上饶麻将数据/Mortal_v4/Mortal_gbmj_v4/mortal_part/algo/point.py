class Point:
    """
    点数计算，根据倍数和底分计算点数，庄家和闲家的点数计算方式相同
    """
    def __init__(self, fan):
        self.fan = fan
        self.ron = fan + 8 * 3
        self.tsumo = (fan + 8) * 3
