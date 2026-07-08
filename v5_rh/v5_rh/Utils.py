from .GameConfig import GameConfig
import lib_MJ as MJ

# 这是工具类，因为循环引用的原因，反正此次

game_config = GameConfig()

def convert_hex2index(a):  # 把对应的十六进制牌转换成[]*34的索引值# 如0x01代表的是第0个数
    if a > 0 and a < 0x10:
        return a - 1
    if a > 0x10 and a < 0x20:
        return a - 8
    if a > 0x20 and a < 0x30:
        return a - 15
    if a > 0x30 and a < 0x40:
        return a - 22

def translate16_33(i):
    """
    16进制转换为10进制
    :param i:
    :return:
    """
    i = int(i)
    if i >= 0x01 and i <= 0x09:
        i = i - 1
    elif i >= 0x11 and i <= 0x19:
        i = i - 8
    elif i >= 0x21 and i <= 0x29:
        i = i - 15
    elif i >= 0x31 and i <= 0x37:
        i = i - 22
    else:
        print("translate16_33 is error,i=%d" % i)
        i = -1
    return i


def trandfer_discards(discards, discards_op, handcards):
    """
    功能：获取场面剩余牌数量，计算手牌和场面牌的数量，再计算未知牌的数量
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param handcards: 手牌
    :return: left_num, discards_list　剩余牌列表，已出现的牌数量列表
    """
    discards_map = {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3, 0x05: 4, 0x06: 5, 0x07: 6, 0x08: 7, 0x09: 8, 0x11: 9, 0x12: 10,
                    0x13: 11, 0x14: 12, 0x15: 13, 0x16: 14, 0x17: 15, 0x18: 16, 0x19: 17, 0x21: 18, 0x22: 19, 0x23: 20,
                    0x24: 21, 0x25: 22, 0x26: 23, 0x27: 24, 0x28: 25, 0x29: 26, 0x31: 27, 0x32: 28, 0x33: 29, 0x34: 30,
                    0x35: 31, 0x36: 32, 0x37: 33, }
    # print ("discards=",discards)
    # print ("discards_op=",discards_op)
    left_num = [4] * 34
    discards_list = [0] * 34
    for per in discards:  # 弃牌数统计
        for item in per:
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:  # 副露数统计
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:  # 手牌统计
        left_num[discards_map[item]] -= 1
    return left_num, discards_list


# 获取ｌｉｓｔ中的最小值和下标
def get_min(list=[]):
    """
    获取最小xts的下标
    :param list: 向听数列表
    :return: 返回最小向听数及其下标
    """
    min = 14
    index = 0
    for i in range(len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min, index

def value_t1(card):
    """
    TODO 用自摸的概率去代表该牌转化成t3的概率？
    计算出牌的危险度评估值，由该牌转化为t3的概率组成  ？不应该是权重吗？
    ？这个函数有什么用没看太懂，是算t1凑成t3的评估值吗？
    :param card:
    :return:
    """
    value = 0
    if card != -1:
        # t1tot3_dict: {'1': [[[1, 1, 1], [1, 1], [1, 6]], [[1, 2, 3], [2, 3], [1, 2]]],
        for e in game_config.t1tot3_dict[str(card)]:
            v = 1
            for i in range(len(e[1])):  # e[1]需要的牌，e[-1]权重
                # 以一万为例,外层循环和内层循环都是两轮:v1=1*T_selfmo[0]*1*T_selfmo[0]*6 v2=1*T_selfmo[1]*1*T_selfmo[2]*2
                v *= game_config.T_SELFMO[MJ.convert_hex2index(e[1][i])] * e[-1][i]
            value += v
        # value = v1 + v2
    return value

def pre_king(king_card=None):
    """
    计算宝牌的前一张，因为宝牌指示牌要翻出来才能知道宝牌是什么
    :param king_card: 宝牌
    :return:宝牌的前一张牌
    """
    if king_card == None:
        return None
    if king_card == 0x01:
        return 0x09
    elif king_card == 0x11:
        return 0x19
    elif king_card == 0x21:
        return 0x29
    elif king_card == 0x31:  # ？上饶麻将游戏规则中，不是东南西北为一个循环，中发白为一个循环吗？
        return 0x37
    else:
        return king_card - 1