import copy
import time
import lib_MJ as MJ  # 使用的一些库函数
from .GameConfig import GameConfig
import logging
import datetime
import itertools
from . import Utils

game_config = GameConfig()
class Node_SSL:
    def __init__(self, take=None, taking_set=[], wan=[], tiao=[], tong=[], zi=[], T1=[], raw=[]):
        self.wan = wan
        self.tiao = tiao
        self.tong = tong
        self.zi = zi
        self.T1 = T1
        self.take = take
        self.raw = raw
        self.taking_set = taking_set
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def node_info(self):
        # print('SSL结点输出-------------')
        print('wan:', self.wan, 'tiao:', self.tiao, 'tong:', self.tong, 'zi:', self.zi, "take:", self.take,
              "taking_set:", self.taking_set)
        # print('+++++++++++++++++++++++')

class ShiSanLan:
    """
    十三烂类
    同花色的牌之间相差三及以上,如一万、四万、七万,7-4=4-1=3
    """
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        """
        类变量初始化
        :param cards:  手牌
        :param suits: 副露
        :param king_card:  宝牌
        :param fei_king: 飞宝数
        :param padding: 填充牌，op操作前，填充-1，使手牌达到14张
        """
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.discard_state = {}  # {'card':[[希望获取的牌的组合],[评估值]]}
        self.discard_score = {}
        self.tree_list = []  # 用于存储所有可能的搜索树的根节点
        self.fei_king = fei_king
        self.padding = padding
        self.test = []
        # 单花色的3张的状态集合
        self.ssl_three_table = [[1, 6, 9],
                                [1, 4, 9],
                                [1, 4, 7],
                                [1, 4, 8],
                                [1, 5, 9],
                                [1, 5, 8],
                                [2, 5, 9],
                                [2, 5, 8],
                                [3, 6, 9],
                                [2, 6, 9]]

        # 单花色的2张的状态集合
        self.ssl_two_table = [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                              [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
                              [3, 6], [3, 7], [3, 8], [3, 9],
                              [4, 7], [4, 8], [4, 9],
                              [5, 8], [5, 9],
                              [6, 9]]
        # 单花色的2张的状态的有效牌集合
        # 和ssl_two_table一一对应
        self.ssl_two_efc = [[[7], [8], [9]], [[8], [9]], [[9]], [[4]], [[4], [5]], [[4], [5], [6]],
                            [[8], [9]], [[9]], [[]], [[5]], [[5], [6]],
                            [[9]], [[]], [[]], [[6]],
                            [[1]], [[1]], [[1]],
                            [[1]], [[1]],  # ？是不是漏了？
                            [[1], [2], [3]]
                            ]

        # 一张
        self.ssl_one_table = [[1], [9], [2], [8], [3], [7], [4], [6], [5]]
        # 一张的有效牌集合
        # 和ssl_one_table一一对应
        self.ssl_one_efc = [[[4, 7], [4, 8], [4, 9], [5, 8], [5, 9], [6, 9]],  # 1
                            [[1, 4], [1, 5], [1, 6], [2, 5], [2, 6], [3, 6]],  # 9
                            [[5, 8], [5, 9], [6, 9], [7], [8]],  # 2  ？为什么有[7]？
                            [[1, 4], [1, 5], [2, 5], [3]],  # 8     ？为什么有[3]?
                            [[6, 9], [7], [8]],  # 3
                            [[1, 4], [2], [3]],  # 7
                            [[1, 7], [8], [9]],  # 4
                            [[1], [2], [3, 9]],  # 6  # ？不是[1], [2]而应该是[1, 9], [2, 9]吧？
                            [[1, 8], [1, 9], [2, 8], [2, 9]]  # 5
                            ]
        # 0张的有效牌集合
        self.ssl_zero_efc = [[1, 6, 9],
                             [1, 4, 9],
                             [1, 4, 7],
                             [1, 4, 8],
                             [1, 5, 9],
                             [1, 5, 8],
                             [2, 5, 9],
                             [2, 5, 8],
                             [3, 6, 9],
                             [2, 6, 9],
                             [2, 7], [3, 7], [3, 8]]

    def add_color(self, list, color):
        """
        给移除花色的烂牌添加花色
        :param list: 烂牌
        :param color: 花色:0x00,0x10,0x20
        :return: list. 添加花色后的牌
        """
        return [i + color for i in list]

    def color_info(self, cards, color):
        """
        计算单花色的相关信息, 包含烂牌[有用牌]和移除烂牌后的无用牌[无用牌]
        :param cards: 单花色的手牌（需要已经移除花色）
        :param color: 花色
        :return: list[[]].[ssl_cards, T1],ssl_cards:烂牌,T1:抽取烂牌后的牌       [[对于13烂有效的牌],[去除有效牌后剩下的牌]]
        """
        CSs = []
        tiles = list(set(cards))  # 在这里去重
        # 计算单花色的有用牌的最大成组牌数,(即是3张成组还是2张成组,后面再算组的数量又是多少)
        waitnumMax1 = max((tiles.count(1) + tiles.count(4) + tiles.count(7)),
                          (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                          (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                          (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                          (tiles.count(1) + tiles.count(5) + tiles.count(9)),
                          (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                          (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                          (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                          (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                          (tiles.count(3) + tiles.count(6) + tiles.count(9)))
        waitnumMax2 = max((tiles.count(2) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(8)),)

        # print('--',max(waitnumMax1, waitnumMax2))
        if max(waitnumMax1, waitnumMax2) == 3:  # 当有用牌数量为3的时候算有多少个不同组
            for tb in self.ssl_three_table:
                if tb[0] in cards and tb[1] in cards and tb[2] in cards:
                    tmp = copy.copy(cards)
                    tmp.remove(tb[0])
                    tmp.remove(tb[1])
                    tmp.remove(tb[2])
                    # 复制cards到tmp,移除tb,tb即为有效牌，剩下的就是剩余牌，一个花色只需要一个组合即可
                    CSs.append([self.add_color(tb, color), self.add_color(tmp, color)])
        elif max(waitnumMax1, waitnumMax2) == 2:  # 当有用牌为2的时候算有多少个不同组
            for i in range(len(self.ssl_two_table)):
                tb = self.ssl_two_table[i]
                if tb[0] in cards and tb[1] in cards:
                    tmp = copy.copy(cards)
                    tmp.remove(tb[0])
                    tmp.remove(tb[1])
                    CSs.append([self.add_color(tb, color), self.add_color(tmp, color)])
        elif max(waitnumMax1, waitnumMax2) == 1:  # 当有用牌只有1的时候
            for card in range(1, 10):
                if card in tiles:
                    tmp = copy.copy(cards)
                    tmp.remove(card)
                    CSs.append([[card + color], self.add_color(tmp, color)])
        else:  # 不可能
            CSs.append([[], []])
        return CSs

    def ssl_CS(self):
        """
        计算十三烂的拆分组合
        计算不同花色的组合信息[[现有的对构成十三烂有效的牌],[无效牌]],再把各花色的信息组合起来
        :return: []，返回拆分结果 [[wan],[tiao],[tong],[zi],[left],xts]
        """
        CSs = []
        if self.suits != []:  # 吃、碰、杠了,不可能再胡13烂了
            return [[[], [], 14]]
        # 花色分离
        wan, tiao, tong, zi = MJ.split_type_s(self.cards)
        # print(wan,tiao,tong,zi)
        # 计算不同花色的组合信息[[现有的对构成十三烂有效的牌],[无效牌]]
        wan_CS = self.color_info(wan, 0)
        tiao_CS = self.color_info([i & 0x0f for i in tiao], 0x10)
        tong_CS = self.color_info([i & 0x0f for i in tong], 0x20)
        # zi
        zi_ssl = list(set(zi))
        zi_T1 = copy.copy(zi)
        # 去除多余的字牌
        for card in zi_ssl:
            zi_T1.remove(card)
            # zi_efc.remove(card)
        zi_CS = [[zi_ssl, zi_T1]]

        # 将各种花色的组合信息全部组合起来
        for cs_wan in wan_CS:
            for cs_tiao in tiao_CS:
                for cs_tong in tong_CS:
                    for cs_zi in zi_CS:
                        xts = 14 - len(cs_wan[0]) - len(cs_tiao[0]) - len(cs_tong[0]) - len(cs_zi[0])
                        CSs.append([cs_wan[0], cs_tiao[0], cs_tong[0], cs_zi[0],
                                    cs_wan[-1] + cs_tiao[-1] + cs_tong[-1] + cs_zi[-1], xts])  # 0有效牌   -1剩余牌
        CSs.sort(key=lambda k: k[-1], reverse=True)  # 将CSs按向听数降序排序
        return CSs

    def efc_ssl(self, cards, type):
        """
        计算后续对十三烂有效的进牌
        :param cards: 单花色的有效牌
        :param type: 花色
        :return: [],合理的最大化有效牌组，例如[1,9] 的 有效牌返回[[4],[5],[6]]
        """
        cards_cp = [i & 0x0f for i in cards]  # 去花色
        if type <= 0x20:  # 非字牌
            if len(cards_cp) == 0:  # 现有有效牌为零张
                efc = self.ssl_zero_efc
            elif len(cards_cp) == 1:
                efc = self.ssl_one_efc[self.ssl_one_table.index(cards_cp)]
            elif len(cards_cp) == 2:
                efc = self.ssl_two_efc[self.ssl_two_table.index(cards_cp)]
            elif len(cards_cp) == 3:  # 有3张无需有效牌
                efc = [[]]
            efc_c = []
            for s in efc:  # 2020.12.28 bug解决，这里没有花色还原
                efc_c.append([i + type for i in s])
            efc = efc_c
        else:  # 字牌的有效牌是从没出现过的字牌
            efc = [[]]
            for card in range(0x31, 0x38):
                if card not in cards:
                    efc[0].append(card)
        return efc

    def expand_node(self, node):
        """
        ssl搜索节点扩展，首先会生成所有可能的摸牌组合，对摸牌组合进行节点的扩展
        :param node: 待扩展的节点
        :return: None
        """
        # 胡牌判断
        if len(node.wan) + len(node.tiao) + len(node.tong) + len(node.zi) == 14:
            # node.node_info()
            return
        else:
            # 与平胡一样。待扩展集合是否为空，不为空直接进行扩展，否则生成待扩展集合，且待扩展集合由现组合的有效牌集构成
            if node.raw != []:
                raw = copy.copy(node.raw)
                card = raw[-1]  # 需要的牌
                raw.pop()
                type = card & 0xf0
                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)
                if type == 0x00:
                    wan = copy.copy(node.wan)
                    wan.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=wan, tiao=node.tiao, tong=node.tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x10:
                    tiao = copy.copy(node.tiao)
                    tiao.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=tiao, tong=node.tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x20:
                    tong = copy.copy(node.tong)
                    tong.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=node.tiao, tong=tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x30:
                    zi = copy.copy(node.zi)
                    zi.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=node.tiao, tong=node.tong,
                                     zi=zi, T1=node.T1, raw=raw)
                node.add_child(child)
                # node.node_info()
                self.expand_node(node=child)
            else:  # 生成raw
                # 对每种花色进行有效牌组合的计算，将他们组合起来，然后生成待扩展的集合
                for wan_efc in self.efc_ssl(node.wan, 0):
                    for tiao_efc in self.efc_ssl(node.tiao, 0x10):
                        for tong_efc in self.efc_ssl(node.tong, 0x20):
                            for zi_efc in self.efc_ssl(node.zi, 0x30):
                                efcs = wan_efc + tiao_efc + tong_efc + zi_efc  # 有效牌集合
                                # print(efcs)
                                xts = 14 - len(node.wan) - len(node.tiao) - len(node.tong) - len(node.zi)
                                for efc in itertools.combinations(efcs, xts):
                                    node.raw = list(efc)  # 有向听数个有效牌
                                    # print(node.raw)
                                    self.test.append(node.raw)
                                    self.expand_node(node)

    def generate_tree(self):
        """
        生成树
        :return:  None，结果保留在类变量中tree_list
        """
        CSs = self.ssl_CS()
        CSs.sort(key=lambda k: k[-1], reverse=True)  # ？这不是降序排序吗？这里应该有问题
        # 取xts最小的一组
        min_xts = CSs[0][-1]
        CSs_min_xts = []
        for cs in CSs:
            if cs[-1] == min_xts:
                CSs_min_xts.append(cs)  # 选取所有向听数最小的组合

        for cs in CSs_min_xts:
            node = Node_SSL(take=None, taking_set=[], wan=cs[0], tiao=cs[1], tong=cs[2], zi=cs[3], T1=cs[4], raw=[])
            self.tree_list.append(node)
            self.expand_node(node=node)

    def cal_score(self, node):
        """
        计算当前节点的评估值
        :param node: 节点
        :return: 评估值
        """

        value = 1
        # print node.taking_set
        # 只需要不重复的单牌,所以这里一个for循环就搞定了
        for card in node.taking_set:  # 且每张牌的权重一样,所以这里只有taking_set,而没有定义taking_set_w
            value *= game_config.T_SELFMO[MJ.convert_hex2index(card)]
        # fan检测
        fan = 8
        # 飞宝
        fei_king = self.fei_king + node.T1.count(self.king_card)
        # 一张宝牌可以使番数乘2
        fan *= 2 ** fei_king
        # 七星
        if len(node.zi) == 7:
            fan *= 2
        score = value * fan
        return score

    def evaluate(self, node):
        """
        胡牌后的节点的评估值计算(也就是整个路径的评估值计算)，此时该路径上的全部弃牌的评估值都设置为一样的
        :param node:
        :return:
        """
        if node.children == []:  # 叶子节点
            if len(node.wan) + len(node.tiao) + len(node.tong) + len(node.zi) == 14:
                score = self.cal_score(node)
                # Node_SSL.node_info(node)
                taking_set_sorted = sorted(node.taking_set)
                discards = node.T1 + self.padding
                for discard in discards:
                    if discard not in self.discard_state.keys():
                        self.discard_state[discard] = [[], []]
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
                    elif taking_set_sorted not in self.discard_state[discard][0]:  # 同一张弃牌可能会有不同希望获取的组合
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
        else:
            for child in node.children:
                self.evaluate(node=child)

    def get_discard_score(self):
        """
        对外总接口，生成所有合理出牌的评估值
        :return:
        """
        self.generate_tree()
        for tree in self.tree_list:
            self.evaluate(node=tree)
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = sum(self.discard_state[discard][-1])
        # print("test",self.test)
        # from collections import Counter
        # element_counts = Counter(map(tuple, self.test))
        # print(element_counts)
        # print(self.discard_state)
        global SSL_DiscardAndTake
        SSL_DiscardAndTake = self.discard_state
        return self.discard_score