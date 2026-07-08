import copy
import time
import lib_MJ as MJ  # 使用的一些库函数
from .GameConfig import GameConfig
import logging
import datetime
import itertools
from . import Utils

game_config = GameConfig()

class JiuYao:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        """
        九幺类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param king_card: 宝牌
        :param fei_king: 飞宝数
        :param padding: 填充，op操作时填充为-1
        """
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.fei_king = fei_king
        self.padding = padding
        self.discard_score = {}
        self.discard_state = {}
        self.yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]

    # ？命名怎么又变幺九了？
    def yaojiu_CS(self):
        """
        生成九幺的拆分组合，不过好像和七对这种对整个牌型有要求的牌型一样，就只有一种组合吧  注：九幺牌型包括国士无双
        :return: [[有效牌],[无效牌],xts]
        """
        # 判断是否是91
        CS = [[], [], 14]
        flag = True
        # 副露中有非幺九牌的话,就不能胡九幺
        for suit in self.suits:
            for card in suit:
                if card not in self.yaojiu:
                    flag = False
        if not flag:
            return CS
        CS[-1] -= len(self.suits) * 3
        for card in self.cards:
            if card in self.yaojiu:
                CS[0].append(card)
            else:
                CS[1].append(card)
        CS[-1] -= len(CS[0])
        return CS

    def get_discard_score(self):
        """
        计算九幺的所有出牌的评估值
        :return: {card:score}，
        """
        CS = self.yaojiu_CS()
        if CS[-1] != 14:
            value = 1
            yaojiu_take = 0
            n = 0
            for card in self.yaojiu:
                # 同一张牌有3张,则再进一张的话可以开杠,能加番且能进进张加快牌型成型，因此权重设置很大
                if CS[0].count(card) > 2 and CS[-1] != 1:  # todo 待完善
                    w = 6
                else:
                    w = 1
                n += game_config.LEFT_NUM[MJ.convert_hex2index(card)]
                yaojiu_take += game_config.T_SELFMO[MJ.convert_hex2index(card)] * w  # todo 重复摸牌的处理
            value *= yaojiu_take ** CS[-1]

            xt = CS[-1]
            j = 0
            while xt > 1:
                j += 1
                if n == 0:  # 若没有剩余牌了,则这步操作会使value变得非常小(负的)
                    n = 0.000001
                value *= float(n - j) / n  # 有剩余牌则会继续调整评估值
                xt -= 1
                if n > 1:
                    n -= 1
            # fan计算
            fan = 4
            fei_king = self.fei_king + CS[1].count(self.king_card)
            fan *= 2 ** fei_king
            if len(self.suits) == 4:  # 单吊
                fan *= 2
            # 有七星91吗？TODO   七星是指东南西北中发白
            score = value * fan
            discards = CS[1] + self.padding
            for discard in discards:
                if discard not in self.discard_score:
                    self.discard_score[discard] = score

        return self.discard_score