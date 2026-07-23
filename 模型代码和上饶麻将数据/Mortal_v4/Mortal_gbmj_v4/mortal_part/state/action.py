"""当前状态的合法动作摘要。

``PlayerState.update`` 会根据规则填充这些布尔字段；特征编码器据此生成
动作 mask，事件回放器据此判断当前是否应该产生监督样本。
"""


class ActionCandidate:
    def __init__(self, target_actor=None):
        self.can_discard = False
        self.can_chi_low = False
        self.can_chi_mid = False
        self.can_chi_high = False
        self.can_pon = False
        self.can_daiminkan = False
        self.can_kakan = False
        self.can_ankan = False
        self.can_tsumo_hu = False
        self.can_ron_hu = False
        self.target_actor = target_actor  # 目标操作者

    @property
    def can_chi(self):
        """
        判断是否可以吃牌
        @return: 如果可以低吃、中吃或者高吃则true
        """
        return self.can_chi_low or self.can_chi_mid or self.can_chi_high

    @property
    def can_kan(self):
        """
        判断是否可以杠牌
        @return: 如果可以明杠或者暗杠或者加杠则true
        """
        return self.can_daiminkan or self.can_kakan or self.can_ankan

    @property
    def can_hu(self):
        """
        判断是否可以胡牌
        @return: 如果可以自摸胡或者荣胡则true
        """
        return self.can_tsumo_hu or self.can_ron_hu

    @property
    def can_pass(self):
        """
        判断是否可以过牌
        @return: 如果可以碰杠或者荣胡则true
        """
        return self.can_chi or self.can_pon or self.can_daiminkan or self.can_ron_hu

    @property
    def can_act(self):
        """
        判断是否可以行动
        @return: 如果可以弃牌、碰、杠、胡则true
        """
        # 只要存在一种可以执行的动作，当前状态就值得加入监督数据集。
        return self.can_discard or self.can_chi or self.can_pon or self.can_kan or self.can_hu

    def __repr__(self) -> str:
        return f"ActionCandidate({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
