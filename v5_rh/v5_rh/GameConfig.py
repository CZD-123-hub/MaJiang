import time


class GameConfig:
    _instance = None  # 类变量，用于存储唯一实例

    def __new__(cls, *args, **kwargs):
        """实现单例模式，保证只创建一个实例"""
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # 避免重复初始化
            # 初始化全局变量
            self.TIME_START = time.time()  # 游戏开始时间
            self.w_type = 0  # lib_MJ的权重选择
            self.ROUND = 0  # 当前轮数
            self.REMAIN_NUM = 136  # 剩余牌数
            self.T_SELFMO = [0] * 34  # 自摸概率表，牌墙中的概率
            self.LEFT_NUM = [0] * 34  # 未出现的牌的数量表
            self.bal_feibao = 1.2
            self.bal = 0
            self.bal_xts = 14
            self.PH_xts = 14
            self.JY_xts = 14
            self.SSL_xts = 14
            self.QD_xts = 14
            self.ROUND = 0
            self.t3Set = None  # 初始化为空，稍后加载
            self.t2Set, self.t2Efc, self.efc_t2index = None, None, None  # 初始化为空
            self.t1tot3_dict = None  # 初始化为空
            self.t2tot3_dict = None  # 初始化为空
            self.initialized = True  # 标记初始化完成
            self.recommended_hu_types_ting = []
            # 延迟导入 lib_MJ
            import lib_MJ as MJ
            self.t3Set = MJ.get_t3info()  # t3的合集
            self.t2Set, self.t2Efc, self.efc_t2index = MJ.get_t2info()  # t2的集合
            self.t1tot3_dict = MJ.t1tot3_info()  # t1转化为t3的字典
            self.t2tot3_dict = MJ.t2tot3_info()  # t2转化为t3的字典
