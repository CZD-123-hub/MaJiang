import copy
import time
import lib_MJ as MJ  # 使用的一些库函数
from .GameConfig import GameConfig
import logging
# import opp_srmj as DFM  # 对手建模
import datetime
import itertools

game_config = GameConfig()


class Node_Qidui:
    def __init__(self, take=None, AA=[], T1=[], raw=[], taking_set=[], king_num=0):
        """
        七对节点变量初始化
        :param take: 摸牌
        :param AA: 对子集合
        :param T1: 单张牌集合
        :param raw: 待扩展集合
        :param taking_set: 已摸牌集合     ？不应该是扩展到下一个节点还需要摸的牌吗？
        :param king_num: 未使用的宝数量
        """
        self.take = take
        self.AA = AA
        self.T1 = T1
        self.raw = raw
        self.taking_set = taking_set
        self.king_num = king_num
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print('AA:', self.AA, 'T1:', self.T1, 'raw:', self.raw, 'taking_set:', self.taking_set, 'king_num',
              self.king_num)


class Qidui:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        """
        七对类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param king_card: 宝牌
        :param fei_king: 飞宝数量
        :param padding: 填充牌，op操作时填充-1 ，一般来说，七对不会有这种操作   ？应该不需要这个参数？
        """
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.fei_king = fei_king
        self.discard_score = {}
        self.king_num = cards.count(king_card)
        self.padding = padding
        self.tree_list = []
        self.discard_state = {}

    def qidui_CS(self):
        """
        计算七对组合的生成
        :return:
        """
        CS = [[], [], 14]  # [[对子],[单张],xts]
        if self.suits != []:  # 有副露,不可能凑成7对
            return CS
        cards_rm_king = copy.copy(self.cards)
        for i in range(self.king_num):
            cards_rm_king.remove(self.king_card)  # 去除宝牌的手牌
        for card in list(set(cards_rm_king)):
            n = cards_rm_king.count(card)
            if n == 1:
                CS[1].append(card)
            elif n == 2:
                CS[0].append([card, card])
            elif n == 3:
                CS[0].append([card, card])
                CS[1].append(card)
            elif n == 4:
                CS[0].append([card, card])
                CS[0].append([card, card])
        king_num = self.king_num
        # 这里把宝用掉
        while king_num > 0:
            if len(CS[0]) + king_num > 7:
                CS[0].append([self.king_card, self.king_card])
                king_num -= 2  # 宝还原
            else:
                CS[0].append([0, 0])
                king_num -= 1   # 宝吊
        CS[-1] -= len(CS[0]) * 2 + (7 - len(CS[0]))
        # CS[-1]+=2  # todo 这里给七对的xts+2，减少后面选择打七对的概率
        if CS[-1] >= 4:  # todo  如果对子的数量过少，不建议打七对
            CS[-1] += 3
        if CS[-1] < 0:
            CS[-1] = 0
        return CS

    def expand_node(self, node):
        """
        节点扩展
        :param node:
        :return:
        """
        # 与平胡类似，若待扩展集合为空则先生成待扩展集合，再进行节点扩展
        if len(node.AA) == 7:  # 胡牌判断
            return
        else:
            if node.raw != []:
                # for card in node.raw:
                card = node.raw[-1]
                node.raw.pop()
                AA = copy.copy(node.AA)
                AA.append([card, card])
                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)
                child = Node_Qidui(take=card, AA=AA, T1=node.T1, raw=node.raw, taking_set=taking_set,
                                   king_num=node.king_num)
                node.add_child(child=child)
                self.expand_node(node=child)
            else:
                if node.T1 != []:
                    t1_sets = copy.copy(node.T1)
                    # if -1 in t1_sets:
                    #     t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    # 计算所有可能的t1替补方案
                    for t1_set in itertools.combinations(t1_sets, min(7 - len(node.AA), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        """
        生成树
        :return:
        """
        CS = self.qidui_CS()
        # print "qidui CS",CS
        node = Node_Qidui(take=None, AA=CS[0], T1=CS[1], taking_set=[], king_num=self.king_num)
        # 为什么这里只有一个节点呢？因为其初始手牌的有效组合只有一种(根据已有对子固定了),需要的有效牌不同,则会产生不同的子节点,但根节点只有一个
        self.tree_list.append(node)
        self.expand_node(node=node)

    def fan(self, node):
        """
        七对番型
        :param node:
        """
        fei_king = self.fei_king + node.T1.count(self.king_card)
        if self.king_num == 0 or fei_king == self.fei_king + self.king_num:
            # 清七对 16分
            fan = 16
        else:
            # ？有宝七对 12分？
            fan = 12
        # 算飞宝番数
        fan *= 2 ** fei_king
        # 91
        jiuyao = [1, 9, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        ziyise = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        flag_jiuyao = True
        for t2 in node.AA:
            if t2[0] not in jiuyao:
                flag_jiuyao = False
                break
        flag_ziyise = True
        for t2 in node.AA:
            if t2[0] not in ziyise:
                flag_ziyise = False
                break
        # 九幺七小对和字一色七小对还要翻倍
        if flag_jiuyao:
            fan *= 2
        if flag_ziyise:
            fan *= 2
        return fan

    def evaluate(self, node):
        """
        胡牌后节点评估值计算(也就是整条路径的评估值计算)
        :param node:
        :return:
        """
        if node.children == []:
            if len(node.AA) == 7:
                # node.node_info()
                taking_set_sorted = sorted(node.taking_set)
                value = 1
                for card in taking_set_sorted:
                    # print "card",card
                    if card == -1:  # ？这里应该不需要填充吧，所以不会有card = -1的情况？还是说这里是宝吊？
                        value = 1.0 / 34
                    else:
                        value *= game_config.T_SELFMO[MJ.convert_hex2index(card)]
                fan = self.fan(node=node)
                # print('QD',fan,value)

                score = value * fan
                discards = node.T1 + self.padding   # ？这里应该不需要填充吧？
                for discard in discards:
                    if discard not in self.discard_state.keys():
                        self.discard_state[discard] = [[], []]
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
                    elif taking_set_sorted not in self.discard_state[discard][0]:
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
        else:
            for child in node.children:
                self.evaluate(child)

    def get_discard_score(self):
        """
        总接口
        生成所有合理出牌的评估值
        :return: {card:score}
        """
        # t1 = time.time()
        self.generate_tree()
        # t2 = time.time()
        for tree in self.tree_list:
            self.evaluate(tree)
        # t3=time.time()
        # print ("qidui time",t2-t1,t3-t2)
        for discard in self.discard_state.keys():
            if discard not in self.discard_score:
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
        global QD_DiscardAndTake
        QD_DiscardAndTake = self.discard_state
        return self.discard_score