import copy
import time
import lib_MJ as MJ  # 使用的一些库函数
from .GameConfig import GameConfig
import logging
import datetime
import itertools
from . import Utils

game_config = GameConfig()


class Node_PH:
    def __init__(self, take=None, AAA=[], ABC=[], jiang=[], T2=[], T1=[], raw=[], taking_set=[], taking_set_w=[],
                 king_num=0,
                 fei_king=0, baohuanyuan=False):
        self.take = take  # 摸牌
        self.AAA = AAA
        self.ABC = ABC
        self.jiang = jiang
        self.T2 = T2
        self.T1 = T1
        self.raw = raw  # 应该表示待扩展手牌吧
        self.king_num = king_num
        self.fei_king = fei_king
        self.children = []
        self.taking_set = taking_set    # ？进牌集合？？take=-1表示宝还原、=0表示用宝牌补、=x(x表示其它牌)表示进张x与t1或t2构成t3？
        self.baohuanyuan = baohuanyuan
        self.taking_set_w = taking_set_w  # ？进牌的权重？

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print('AAA=', self.AAA, 'ABC=', self.ABC, 'jiang=', self.jiang, "T1=", self.T1, "T2=", self.T2, 'raw=',
              self.raw,
              'taking_set=', self.taking_set, 'king_num=', self.king_num, 'fei_king=', self.fei_king, 'baohuanyuan=',
              self.baohuanyuan)


class SearchTree_PH():
    """
    平胡搜索模块
    """
    def __init__(self, hand, suits, combination_sets, king_card=None, fei_king=0):
        """
        类变量初始化
        :param hand: 手牌
        :param suits: 副露
        :param combination_sets: 拆分组合集合
        :param king_card: 宝牌
        :param fei_king: 飞宝数
        """
        self.hand = hand
        self.suits = suits
        self.combination_sets = combination_sets
        self.xts = combination_sets[0][-2]
        self.tree_dict = []  # 存储树的字典
        self.king_card = king_card
        self.fei_king = fei_king
        if king_card != None:
            self.king_num = hand.count(king_card)
        else:
            self.king_num = 0
        self.discard_score = {}  # 出牌集合的评估值集合,和discard_state相对应
        self.discard_state = {}  # 出牌集合的状态集合 字典 {'card':[[希望获取的牌],[获取每个牌的评估值]]}
        self.node_num = 0  # 统计节点数目（观测值）
        self.chang_num = 0  # 统计状态不同但分数相同的节点 （观测值）

    def expand_node(self, node):
        """
        节点扩展.首先扩展将牌，再扩展t3:先扩展t2->t3,再t1->t3,使用itertools.combinations生成待扩展集合可以有效减少重复计算量
        :param node:
        :return: None
        一、先扩展将牌：
        1.有两张宝牌,两张宝牌直接充当将牌,即宝还原.
        2.已有宝牌，差任一张牌凑成将牌.
        3.t2中有对子,将对子作为将.
        4.无宝牌、无对子,使用t1或t2进行扩展.
        二、再判断是否胡牌：有将后,判断当前情况是否满足胡牌条件,并弃掉多余的宝牌.
        三、再扩展t3节点：
        1.若待扩展集合不空：判断集合中是t2还是t1.
        ①若是t2,则搜索t2tot3的状态转换表,找到有效牌,然后根据宝牌的有无进行宝还原、宝吊和正常扩展.
        ②若是t1,则搜索t1tot3的状态转换表,找到有效牌,然后根据宝牌的有无进行宝还原、宝吊、宝牌补一张和正常扩展.
        2.若待扩展集合为空：则使用t2或t1进行t3节点扩展
        """
        # node.node_info()
        # 先定将
        if node.jiang == []:
            has_jiang = False
            if node.king_num >= 2:  # 宝还原
                has_jiang = True  # 有宝还原时不再搜索无将情况
                child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=[self.king_card, self.king_card],
                                T2=node.T2,
                                T1=node.T1,
                                taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                king_num=node.king_num - 2,
                                fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                node.add_child(child=child)
                self.expand_node(child)


            if node.king_num > 0:  # 宝吊
                has_jiang = True  # 宝吊不再搜索无将,因为一张宝牌可以和任意一张牌凑成将牌
                taking_set = copy.copy(node.taking_set)
                taking_set.append(0)  # 填充0,表示进张为宝牌？
                taking_set_w = copy.copy(node.taking_set_w)
                taking_set_w.append(1)  # 填充1,表示宝牌的权重为1？
                child = Node_PH(take=0, AAA=node.AAA, ABC=node.ABC, jiang=[0, 0], T2=node.T2,
                                T1=node.T1,
                                taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num - 1,
                                fei_king=node.fei_king, baohuanyuan=False)
                node.add_child(child=child)
                self.expand_node(child)

            if node.king_num <= 1:  # 有1张或0张宝牌且有将牌时扩展
                for t2 in node.T2:  # 有将
                    T2 = MJ.deepcopy(node.T2)
                    # 从t2中找到对子作为将牌
                    if t2[0] == t2[1]:
                        has_jiang = True  # 有将不再搜索无将
                        T2.remove(t2)
                        child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=t2, T2=T2,
                                        T1=node.T1,
                                        taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                        king_num=node.king_num,
                                        fei_king=node.fei_king, baohuanyuan=False)  # 非宝吊、宝还原
                        node.add_child(child=child)
                        self.expand_node(node=child)

            # 上面都尝试过但还是无宝牌、无将的情况，用T1或T2进行扩展
            if not has_jiang:  # todo 这里可以考虑有将时也扩展
                jiangs = copy.copy(node.T1)
                # 没有T1,则选择一个t2来扩展
                # todo 可以在有T1时也扩展该部分
                if jiangs == []:
                    for t2 in node.T2:
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:
                            taking_set = copy.copy(node.taking_set)
                            taking_set.append(t1)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.append(1)
                            T1 = copy.copy(jiangs)
                            T1.remove(t1)
                            child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=T2, T1=T1,
                                            taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num,
                                            fei_king=node.fei_king, baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                # 有T1，则从T1中选择一张作为将
                else:
                    for t1 in jiangs:
                        if t1 == -1:  # op填充的-1不作扩展，应该是在推荐动作时当手牌不满14张时填充的？
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(1)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=node.T2,
                                        T1=T1,
                                        taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num,
                                        fei_king=node.fei_king, baohuanyuan=False)
                        node.add_child(child=child)
                        self.expand_node(node=child)

        # 胡牌判断,存在将牌的前提下去判断是否胡牌
        elif len(node.AAA) + len(node.ABC) == 4:
            if node.king_num > 0:
                node.fei_king += node.king_num  # 多余的宝牌没使用，直接飞宝
                node.king_num = 0
                if node.baohuanyuan and node.fei_king == self.king_num + self.fei_king:  # 宝牌全部飞完了，所以就不是宝还原了
                    node.baohuanyuan = False
            return

        # T3扩展
        else:
            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw != []:
                tn = node.raw[-1]
                raw = copy.copy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                # tn是列表,则使用t2扩展t3
                if type(tn) == list:
                    t2 = tn
                    # t2tot3_dict: {'[1, 1]': [[[1, 1], [1, 1, 1], [], 1, 6]]
                    # ‘[card,card]’:[[[card,card],[3张],[手牌剩余牌],有效牌,有效牌权重],[···]]
                    for item in game_config.t2tot3_dict[str(t2)]:
                        if item[1][0] == item[1][1]:  # t2凑AAA
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[1])
                            ABC = node.ABC
                        else:  # t2凑ABC
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[1])
                        if node.king_num > 0 and item[-2] == self.king_card:  # 宝还原,需要的牌和宝牌一样
                            child = Node_PH(take=-1, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=node.taking_set,
                                            taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 1,
                                            fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num > 0 and (0 in node.jiang):  # 宝牌补一张
                            child = Node_PH(take=0, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=node.taking_set,
                                            taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 1,
                                            fei_king=node.fei_king, baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                        else:  # 没有宝牌了,正常打法
                            taking_set = copy.copy(node.taking_set)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set.append(item[-2])
                            taking_set_w.append(item[-1])
                            child = Node_PH(take=item[-2], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                            king_num=node.king_num,
                                            fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                # 若tn是int整型,则t1扩展为t3
                elif type(tn) == int:
                    t1 = tn
                    for item in game_config.t1tot3_dict[str(t1)]:
                        # t1tot3_dict: {'1': [[[1, 1, 1], [1, 1], [1, 6]], [[1, 2, 3], [2, 3], [1, 2]]],
                        # '2': [[[1, 2, 3], [1, 3], [1, 2]], [[2, 2, 2], [2, 2], [1, 6]], [[2, 3, 4], [3, 4],[1, 2]]]
                        # ‘card’:[[[3张],[有效牌1,有效牌2],[有效牌权重1,有效牌权重2]],[···]]
                        flag2 = False
                        if node.king_num > 0:  # 用于处理宝还原
                            for card in item[1]:
                                if card == self.king_card:
                                    flag2 = True  # 宝还原标识
                                    raw_copy = copy.copy(raw)
                                    raw_copy.append(sorted([card, t1]))  # t1先扩展为t2
                                    child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=node.jiang, T2=node.T2,
                                                    T1=node.T1, raw=raw_copy,
                                                    taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                                    king_num=node.king_num - 1, fei_king=node.fei_king,
                                                    baohuanyuan=node.baohuanyuan)
                                    node.add_child(child=child)
                                    self.expand_node(node=child)
                        if flag2:  # 上述宝还原后不再继续扩展
                            continue

                        if item[0][0] == item[0][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[0])
                            ABC = node.ABC
                        else:
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[0])

                        if node.king_num >= 2:  # 宝牌有2张以上，直接补2张，即使其中有一张被作为宝还原也不影响
                            child = Node_PH(take=[0, 0], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                            raw=raw,
                                            taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 2, fei_king=node.fei_king,
                                            baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num == 0:  # 宝数量为0时正常处理
                            take = item[1]
                            take_w = item[-1]

                            taking_set = copy.copy(node.taking_set)
                            taking_set.extend(take)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.extend(take_w)
                            child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                            raw=raw,
                                            taking_set=taking_set, taking_set_w=taking_set_w,
                                            king_num=node.king_num, fei_king=node.fei_king,
                                            baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num == 1:  # king_num=1 ,补一张牌
                            # 用1张宝牌,可扩展两个节点
                            for i in range(len(item[1])):
                                card = item[1][i]
                                take = [0, card]

                                taking_set = copy.copy(node.taking_set)
                                taking_set.append(card)
                                taking_set_w = copy.copy(node.taking_set_w)
                                taking_set_w.append(1)

                                child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                                raw=raw,
                                                taking_set=taking_set, taking_set_w=taking_set_w,
                                                king_num=node.king_num - 1, fei_king=node.fei_king,
                                                baohuanyuan=False)
                                node.add_child(child=child)
                                self.expand_node(node=child)
                        else:
                            pass
                            # logger.error("node.king_num==%s", (node.king_num))
                else:
                    pass
                    # logger.error("tn Error")
            else:  # 若待扩展集合为空则生成待扩展集合
                if node.T2 != []:  # 1、先扩展T2为T3
                    t2_sets = node.T2
                    T2 = copy.copy(node.T2)
                    # 生成待扩展集合   若T2富余,则选择4-len(T3)个T2用于拓展即可;若T2不足,则扩展len(T2)个
                    for t2_set in itertools.combinations(t2_sets, min(4 - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        node.T2 = copy.copy(T2)  # ？这里的写法会不会有问题，和上面重复了？
                        node.raw = list(t2_set)
                        for t2 in node.raw:
                            node.T2.remove(t2)
                        self.expand_node(node=node)
                #  生成T1扩展T3集合
                elif node.T1 != []:
                    t1_sets = copy.copy(node.T1)
                    # 这里移除了填充的-1，不作扩展
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    # 若T3>=4,则无需拓展;若T3不足,则要根据T3和单张的数量来拓展
                    for t1_set in itertools.combinations(t1_sets, min(4 - len(node.AAA) - len(node.ABC), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        """
        构建生成树的根节点，并沿根节点扩展到一整棵生成树
        :return: None
        """
        kz = []
        sz = []
        # 将副露连同手牌的T3一起加入到节点的AAA和ABC状态中
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append(t3)
            else:
                sz.append(t3)
        # 使用拆分组合生成树
        for cs in self.combination_sets:
            root = Node_PH(take=None, AAA=cs[0] + kz, ABC=cs[1] + sz, jiang=[], T2=cs[2] + cs[3], T1=cs[-1],
                           taking_set=[], taking_set_w=[], king_num=self.king_num,
                           fei_king=self.fei_king, baohuanyuan=self.king_num > 0)
            # 每一棵树都存储到树集合中，并从树根节点开始扩展
            self.tree_dict.append(root)
            self.expand_node(node=root)

    def cal_score(self, node):
        """
        节点评估值计算模块：该节点评估值即基础价值x自摸概率×进牌权重×番数
        :param node:
        :return: float 评估值
        """
        value = 1
        if node.taking_set_w != []:
            # 上饶麻将中胡牌需要自摸，获取权重为1。这里将具有最小获取权重的牌的权重置为1.是一种权重最大化的处理.todo 可以尝试其他的处理
            node.taking_set_w[node.taking_set_w.index(min(node.taking_set_w))] = 1
            for i in range(len(node.taking_set)):
                card = node.taking_set[i]
                if card == 0:  # 宝吊的任意牌,获取概率为1
                    taking_rate = 1.0
                else:  # 其他牌的获取概率计算
                    taking_rate = game_config.T_SELFMO[MJ.convert_hex2index(card)]
                value *= taking_rate * node.taking_set_w[i]  # todo 需要结合其他玩家打出这张牌的概率来计算，将获取权重具体化

        # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例   ？为什么这么修改？
        taking_set = list(set(node.taking_set))
        taking_set_num = [node.taking_set.count(i) for i in taking_set]
        for i in range(len(taking_set_num)):
            n = taking_set_num[i]
            j = 0
            while n > 1:  # 即需要的n超过1才会进入这个循环,因为上面的部分已经处理过需要一张牌的情况
                j += 1
                index = MJ.convert_hex2index(taking_set[i])
                if game_config.LEFT_NUM[index] >= n:  # 当前还剩余牌数/总剩余牌数
                    value *= float(game_config.LEFT_NUM[index] - j) / game_config.LEFT_NUM[index]
                else:  # 需要摸的牌数超过了剩余的牌数，直接舍弃
                    value = 0
                    return value
                n -= 1

        # 番数计算
        fan = Fan_PH(kz=node.AAA, sz=node.ABC, jiang=node.jiang, fei_king=node.fei_king,
                     using_king=self.king_num + self.fei_king - node.fei_king,
                     baohuanyuan=node.baohuanyuan).fanDetect()
        # 单吊翻倍
        if len(self.suits) == 4:
            fan *= 2
        score = fan * value
        # print('PH',fan,value)
        return score

    def calculate_path_expectation(self, node):
        """
        计算整条路径的上的评估值，并将其赋予为所有出牌的评估值
        :param node:
        :return:
        """
        # 深度搜索。搜索胡牌的叶子节点
        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:
            self.node_num += 1
            discard_set = []

            # 计算出牌集合（保持原逻辑）
            for i in range(node.fei_king - self.fei_king):
                discard_set.append(self.king_card)
                break
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)

            if not discard_set:
                return

            taking_set_sorted = sorted(node.taking_set)
            taking_set_label = str(taking_set_sorted)
            score = self.cal_score(node=node)

            # 优化后的处理逻辑
            unique_discard_set = set(discard_set)
            for card in unique_discard_set:
                if card not in self.discard_state:
                    self.discard_state[card] = {}

                # 直接使用字典操作，时间复杂度 O(1)
                if taking_set_label not in self.discard_state[card]:
                    self.discard_state[card][taking_set_label] = score
                else:
                    if score > self.discard_state[card][taking_set_label]:
                        self.chang_num += 1
                        self.discard_state[card][taking_set_label] = score

        elif node.children:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        """
            总接口。获取出牌的评估值
            :return: dict. 出牌的评估值集合
        """
        self.discard_score = {}  # 出牌集合的评估值集合
        self.discard_state = {}  # 出牌集合的状态集合  {card：[[出card后希望获取的组合],[获取每个组合的评估值]]}
        # t1 = time.time()
        self.generate_tree()
        # t2 = time.time()
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        # t3 = time.time()
        for card, score_dict in self.discard_state.items():
            self.discard_score[card] = sum(score_dict.values())
        return self.discard_score

'''
番数计算类  
'''
class Fan_PH():
    def __init__(self, kz, sz, jiang, fei_king=0, using_king=0, baohuanyuan=False):
        """
        初始化类变量
        :param kz: 刻子
        :param sz: 顺子
        :param jiang: 将
        :param node: 待检测的结点
        :param fei_king: 飞宝数
        """
        self.kz = kz
        self.sz = sz
        self.jiang = jiang
        self.fei_king = fei_king
        self.using_king = using_king
        self.baohuanyuan = baohuanyuan
        self.mul = 2  # 用于番数翻倍

    # 碰碰胡
    def pengPengHu(self):
        """
        碰碰胡检测
        是否刻子树数达到４个
        :return: bool
        """
        if len(self.kz) == 4:
            # if self.usingKing==0:
            return True
        else:
            return False

    # 宝还原 x2
    # def baoHuanYuan(self):
    #
    #     if self.baohuanyuan:
    #         return True
    #     else:
    #         return False

    # 清一色 x2
    def qingYiSe(self):
        """
        清一色检测，全是万或条或筒，无字牌
        手牌为同一花色
        :return: bool
        """
        # todo 宝吊无法检测清一色，因为将牌无法确定
        w = 0
        ti = 0
        to = 0
        z = 0  # 可删去
        # print self.kz + self.sz+ self.jiang
        for t in self.kz + self.sz + [self.jiang]:
            card = t[0]
            if card != 0:
                if card & 0xf0 == 0x00:
                    w = 1
                elif card & 0xf0 == 0x10:
                    ti = 1
                elif card & 0xf0 == 0x20:
                    to = 1
                else:
                    return False

        if w + ti + to <= 1:
            return True
        else:
            return False

    def fanDetect(self):
        """
        番数计算
        基础分４分，通过调用上述的番种检测来增加基础分
        :return: int 番数
        """
        # 基础分判定
        score = 4
        if self.pengPengHu():
            # print "0"
            score *= self.mul
            if self.using_king == 0 or self.baohuanyuan:
                score *= self.mul
            score *= 2  # 碰碰胡再给2倍分

        # 翻倍机制
        # 飞宝 当可以宝吊时，将飞宝倍数得到提高
        # if 0 in self.jiang:
        #     for i in range(self.fei_king):
        #         score *= 2.5
        # else:
        for i in range(self.fei_king):
            # print "1"
            score *= self.mul

        # # 宝还原　x2
        if self.baohuanyuan:
            # print score, self.baohuanyuan,self.jiang,
            # print "2"
            score *= self.mul

        # 清一色
        if self.qingYiSe():
            score *= self.mul
            # print "3"
        # 单吊　x2
        # if len
        # 这里无法处理，宝吊需要吃碰杠处理
        # if score>16: #得分大于16时，分数评估提高
        #     score*=1.5
        # print
        return score


'''
平胡类，相关处理方法
分为手牌拆分模块sys_info，评估cost,出牌决策，吃碰杠决策等部分
'''


class PingHu:
    '''
    平胡类模块
    该模块的弃牌评估值计算独立放在Search_PH中了
    '''

    def __init__(self, cards, suits, kingCard=None, fei_king=0, padding=[]):
        """
        类变量初始化
        :param cards: 手牌　
        :param suits:副露
        :param leftNum:剩余牌数量列表
        :param discards:弃牌
        :param discards_real:实际弃牌
        :param discardsOp:场面副露
        :param round:轮数
        :param remainNum:牌墙剩余牌数量
        :param seat_id:座位号
        :param kingCard:宝牌
        :param fei_king:飞宝数
        :param op_card:动作操作牌
        """
        cards.sort()
        self.cards = cards
        self.suits = suits
        self.kingCard = kingCard
        self.fei_king = fei_king
        self.padding = padding
        self.kingNum = cards.count(kingCard)

    @staticmethod
    def split_type_s(cards=[]):
        """
        功能：手牌花色分离，将手牌分离成万条筒字各色后输出
        :param cards: 手牌　[]
        :return: 万,条,筒,字　[],[],[],[]
        """
        cards_wan = []
        cards_tiao = []
        cards_tong = []
        cards_zi = []
        for card in cards:
            if card & 0xF0 == 0x00:
                cards_wan.append(card)
            elif card & 0xF0 == 0x10:
                cards_tiao.append(card)
            elif card & 0xF0 == 0x20:
                cards_tong.append(card)
            elif card & 0xF0 == 0x30:
                cards_zi.append(card)
        return cards_wan, cards_tiao, cards_tong, cards_zi

    @staticmethod
    def get_effective_cards(dz_set=[]):
        """
        获取有效牌
        :param dz_set: 搭子集合 list [[]]
        :return: 有效牌 list []
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
        effective_cards = set(effective_cards)  # set 和list的区别？
        return list(effective_cards)

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中
    @staticmethod
    def in_cards(t32=[], cards=[]):
        """
        判断３２Ｎ是否存在于ｃａｒｄｓ中
        :param t32: ３Ｎ或2N组合牌
        :param cards: 本次判断的手牌
        :return: bool
        """
        for card in t32:
            if card not in cards:
                return False
        return True

    @staticmethod
    def get_32N(cards=[]):
        """
        功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
        思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        :param cards: 手牌　[]
        :return: 3N与2N的集合　[[]]
        """
        cards.sort()
        kz = []
        sz = []
        aa = []
        ab = []
        ac = []
        lastCard = 0
        # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        if True:
            for card in cards:
                if card == lastCard:
                    continue
                else:
                    lastCard = card
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                if cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])
                else:
                    if card + 1 in cards:
                        ab.append([card, card + 1])
                    if card + 2 in cards:
                        ac.append([card, card + 2])
        return kz + sz + aa + ab + ac

    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        """
        功能：递归计算手牌的所有组合信息，并存储在t32_set，
        思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
        :param cards: 手牌
        :param t32_branch: 本次递归的暂存结果
        :param t32_set: 所有组合信息
        :return: 结果存在t32_set中
        """
        t32N = self.get_32N(cards=cards)
        # print(t32N)

        if len(t32N) == 0:
            t32_set.extend(t32_branch)
            # t32_set.extend([cards])
            t32_set.append(0)
            t32_set.extend([cards])
        else:
            for t32 in t32N:
                if self.in_cards(t32=t32, cards=cards):
                    cards_r = copy.copy(cards)
                    for card in t32:
                        cards_r.remove(card)
                    t32_branch.append(t32)
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)

    def tree_expand(self, cards):
        """
        功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
        思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况
        :param cards: cards [] 万条筒其中一种花色手牌
        :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
        """
        all = []
        t32_set = []
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
        # print("t32",t32_set)
        kz = []
        sz = []
        t2N = []
        aa = []
        length_t32_set = len(t32_set)
        i = 0
        # for i in range(len(t32_set)):
        while i < length_t32_set:
            t = t32_set[i]
            flag = True  # 本次划分是否合理
            if t != 0:
                if len(t) == 3:
                    if t[0] == t[1]:
                        kz.append(t)
                    else:
                        sz.append(t)  # print (sub)
                elif len(t) == 2:
                    if t[1] == t[0]:
                        aa.append(t)
                    else:
                        t2N.append(t)

            else:
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
                # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
                for card in leftCards:
                    if card in efc_cards:
                        flag = False
                        break

                if flag:
                    all.append([kz, sz, aa, t2N, 0, leftCards])
                kz = []
                sz = []
                aa = []
                t2N = []
                i += 1  # 跳过向听数
            i += 1  # 跳过剩余手牌

        allSort = []  # 给每一个元素排序
        allDeWeight = []  # 排序去重后

        for e in all:
            for f in e:
                if f == 0:  # 0是xts位，int不能排序
                    continue
                else:
                    f.sort()
            allSort.append(e)
        # print("去重前：", allSort)
        for a in allSort:
            if a not in allDeWeight:
                allDeWeight.append(a)
        # print("去重后：", allDeWeight)
        allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)
        return allDeWeight

    @staticmethod
    def zi_expand(cards=[]):
        """
        功能：计算字牌组合信息
        思路：字牌组合信息需要单独计算，因为没有字顺子，迭代计算出各张字牌的２Ｎ和３Ｎ的情况，由于某些情况下，可能只会需要ａａ作为将牌的情况，同时需要刻子和ａａ的划分结果
        :param cards: 字牌手牌
        :return: ziBranch　字牌的划分情况　[kz,sz,aa,ab,xts,leftCards]
        """
        cardList = []
        for i in range(7):
            cardList.append([])
        ziCards = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        for card in ziCards:
            index = (card & 0x0f) - 1
            # print(index)
            if cards.count(card) == 4:
                # 此结构为[3N,2N,leftCards]
                cardList[index].append([[[card, card, card]], [], [], [], 0, [card]])
            elif cards.count(card) == 3:
                cardList[index].append([[[card, card, card]], [], [], [], 0, []])
                cardList[index].append([[], [], [[card, card]], [], 0, [card]])
            elif cards.count(card) == 2:
                cardList[index].append([[], [], [[card, card]], [], 0, []])
            elif cards.count(card) == 1:
                cardList[index].append([[], [], [], [], 0, [card]])
            else:
                cardList[index].append([[], [], [], [], 0, []])

        ziBranch = []
        for c1 in cardList[0]:
            for c2 in cardList[1]:
                for c3 in cardList[2]:
                    for c4 in cardList[3]:
                        for c5 in cardList[4]:
                            for c6 in cardList[5]:
                                for c7 in cardList[6]:
                                    branch = []
                                    for n in range(6):
                                        branch.append(c1[n] + c2[n] + c3[n] + c4[n] + c5[n] + c6[n] + c7[n])
                                    ziBranch.append(branch)
        return ziBranch

    def pinghu_CS2(self, cards=[], suits=[], t1=[]):
        """
        功能：综合计算手牌的组合信息，并计算向听数和选择向听数最少的组合返回
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息，再将其综合起来，计算每个组合向听数，最后输出向听数最小的组合
        :param cards: 手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param kingCard: 宝牌
        :return: 返回向听数最少的组合信息
        """
        # 去除宝牌计算信息，后面出牌和动作决策再单独考虑宝牌信息
        if cards == []:
            cards = self.cards
            suits = self.suits
        RM_King = copy.copy(cards)
        kingNum = 0
        if self.kingCard != None:
            kingNum = cards.count(self.kingCard)
            for i in range(kingNum):
                RM_King.remove(self.kingCard)

        # 花色分离
        wan, tiao, tong, zi = self.split_type_s(RM_King)
        wan_expd = self.tree_expand(cards=wan)
        tiao_expd = self.tree_expand(cards=tiao)
        tong_expd = self.tree_expand(cards=tong)
        zi_expd = self.zi_expand(cards=zi)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                    for m in zi_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n] + m[n])

                        branch[-1] += self.padding + t1
                        all.append(branch)

        # 将获取概率为０的组合直接丢弃到废牌中 todo 由于有宝，这里也可能会被宝代替
        # 移到了出牌决策部分处理
        if self.kingNum <= 1:  # 这里只考虑正常出牌、宝做宝吊的情况(宝吊:听牌时听的是宝牌),所以不考虑宝牌作万能牌的情况
            for a in all:
                for i in range(len(a[3]) - 1, -1, -1):  # 倒序遍历,(起始值,终止值,步长),包含两个端点
                    ab = a[3][i]
                    efc = self.get_effective_cards([ab])
                    if sum([game_config.LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) == 0:
                        # 有效牌已经全部出现,该搭子不可能再凑成3张,将该搭子加入单张集
                        a[3].remove(ab)
                        a[-1].extend(ab)
                        # logger.info("remove rate 0 ab,%s,%s,%s,a=%s",self.cards,self.suits,self.kingCard,a)

        # 计算向听数
        # 计算拆分组合的向听数
        all = MJ.cal_xts_2(all, suits, kingNum)
        # print('all:',all)
        return all
        # # 获取向听数最小的all分支
        # min_index = 0
        # for i in range(len(all)):
        #     if all[i][4] > all[0][4]:  # xts+1以下的组合(不包括xts+1)
        #         min_index = i
        #         break
        #
        # if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予all长度
        #     min_index = len(all)
        #
        # all = all[:min_index]
        #
        # # ？向听数为0不是胡牌了吗？
        # # 选取好向听数最少的组合时,若其向听数为0,则处理向听数为0时的情况,
        # # 需要从中依次选择一张牌作为t1,重新组成向听数为1的组合集合
        # if all[0][-2] == 0 and all[0][-1] == []:
        #     all = []
        #     for card in list(set(cards)):
        #         cards_ = copy.copy(cards)
        #         cards_.remove(card)
        #         all += self.pinghu_CS2(cards=cards_, suits=suits, t1=[card])
        # return all

    def left_card_weight(self, card, left_num, need_jiang=False):
        """
        功能：对废牌组合中的每张废牌进行评估，计算其成为３Ｎ的概率  (？应该是权重或者说评估值吧？)
        思路：每张牌能成为３N的情况可以分为先成为搭子，在成为３Ｎ２步，成为搭子的牌必须自己摸到，而成为kz,sz可以通过吃碰。刻子为获取２张相同的牌，顺子为其邻近的２张牌
        :param card: 孤张牌
        :param left_num: 剩余牌数量列表
        :return: 评估值  （？或者说权重？）
        """

        # if self.remainNum==0:
        #     remainNum=1
        # else:
        #     remainNum = self.remainNum
        # remainNum = 1
        i = Utils.convert_hex2index(card)

        if need_jiang:
            return left_num[i]
        # d_w = 0

        # if left_num[i] == self.remainNum:
        #     sf = float(self.leftNum[i])
        # else:
        #     sf = float(left_num[i]) / remainNum * float((left_num[i] - 1)) / remainNum * 6

        # 刻子的权重是4,顺子的权重是2
        # 所有花色均可组合成刻子，所以这里统一计算aa的权重
        if left_num[i] > 1:
            aa = left_num[i] * (left_num[i] - 1) * 4
        else:
            aa = left_num[i]
        if card >= 0x31:  # 字牌的kz权重
            # todo if card == fengwei:
            # if card >= 0x35 and left_num[i] >= 2:
            #     d_w = left_num[i] * left_num[i] * 2  # bug 7.22 修正dw-d_w
            # else:
            d_w = aa  # 7.22 １６:３５ 去除字牌
        elif card % 16 == 1:  # 可组成111 123
            d_w = aa + left_num[i + 1] * left_num[i + 2] * 2
        elif card % 16 == 2:  # 可组成222 123 234
            d_w = aa + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * left_num[i + 2] * 2
        elif card % 16 == 8:  # 可组成888 789 678
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2
        elif card % 16 == 9:  # 可组成999 789
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2
        # 删除多添加的×２
        else:  # 比如说5可以组成555 345 456 567
            # print (left_num)
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * \
                  left_num[i + 2] * 2
        # if card<=0x31:
        #     if (card%0x0f==3 or card %0x0f==7): #给金3银7倍数
        #         d_w*=1.5
        #     elif card%0x0f==5:
        #         d_w*=1.2
        print("i=", i, d_w)
        return d_w

# cards=[1,2,3,6,6,9,9,0x11,0x11]
# ph= PingHu(cards=cards, suits=[])
# print(ph.pinghu_CS2())