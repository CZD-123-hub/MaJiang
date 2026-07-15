import copy
import time
from .get_score_dict import get_score_dict
import lib_MJ as MJ  # 使用的一些库函数
from .GameConfig import GameConfig
from . import Utils


game_config = GameConfig()

def recommend_card(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=136,
                   round=0, seat_id=0):
    """
    功能：推荐出牌接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类出牌决策
    :param cards:
    :param suits: 自己的副露手牌
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param fei_king: 飞宝数
    :param remain_num: 剩余牌
    :return: outCard 推荐出牌
    """
    # logger.info("recommond card start...")
    # 更新全局变量
    ##T_SELFMO:自摸表  LEFT_NUM:剩余表    RT1：其他玩家的状态表[不要,需要]     RT2：[吃碰概率，危险度]

    game_config.ROUND = round  # 轮数
    # print("round",round)
    MJ.KING = king_card  # 宝牌ID
    TIME_START = time.time()
    # 计算获取概率
    #未减去自己的副露。。减了-
    game_config.LEFT_NUM, _ = Utils.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    game_config.LEFT_NUM[Utils.translate16_33(Utils.pre_king(king_card))] -= 1   # 因为宝牌指示牌要翻出来才能知道宝牌是什么，所以剩余数目减1
    game_config.REMAIN_NUM = max(1, min(sum(game_config.LEFT_NUM), remain_num))  # 剩余牌的总数
    if True:
        # 自摸表初始化，某牌的自摸概率为：该牌剩余数/总剩余牌数
        # if round <8:
        game_config.T_SELFMO = [float(i) / game_config.REMAIN_NUM for i in game_config.LEFT_NUM]
        # print(T_SELFMO)
        RT1 = []
        RT2 = []
        RT3 = []
    # else:
    #     # 当round>=8时，使用对手建模
    #     # cards, suits, king_card, fei_king, discards, discardsOp, discardsReal, round, seat_id, xts_round, M
    #     _, T_SELFMO, RT1, RT2, RT3 = DFM.DefendModel(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king,
    #                                                  discards=discards, discardsOp=discards_op, discardsReal=discards,
    #                                                  round=round, seat_id=seat_id, xts_round=DFM.xts_round,
    #                                                  M=100).getWTandRT()
    #     RT1 = []
    #     RT2 = []
    #     RT3 = []
    # t1tot2_dict = MJ.t1tot2_info(T_selfmo=T_SELFMO)

    # t1tot3_dict = MJ.t1tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])
    # t2tot3_dict = MJ.t2tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])

    # 计算四个牌型中xts最少牌型的出牌评估值
    score_dict, _ = get_score_dict(cards, suits, king_card, fei_king)
    # print("score_dict", score_dict)
    if score_dict != {}:
        # score_dict_max = score_dict[max(score_dict, key=lambda x: score_dict[x])]
        recommend_card = max(score_dict, key=lambda x: score_dict[x])  # 选择评估值最大的弃牌作为推荐弃牌
        # print("max_value=", score_dict[recommend_card])
    else:  # 手牌可能已经胡了，这里出一张牌，一般不可能发生
        recommend_card = cards[-1]
        # logger.error("no card be recommonded,cards=%s,suits=%s,king_card=%s",cards, suits, king_card)
    end = time.time()
    if end - TIME_START > 3:  # 超时输出
        pass
    #     logger.error("overtime %s,%s,%s,%s", end - TIME_START, cards, suits, king_card)
    # logger.info("recommend_card %s",recommend_card)
    return recommend_card



def recommend_op(op_card, cards=[], suits=[], king_card=None, discards=[], discards_op=[], canchi=False,
                 self_turn=False, fei_king=0, isHu=False, round=0,seat_id=0, out_seat_id=0):
    """
    功能：动作决策接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类动作决策
    :param op_card: 操作牌
    :param cards: 手牌
    :param suits: 副露
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param canchi: 吃牌权限
    :param self_turn: 是否是自己回合
    :param fei_king: 飞宝数
    :param isHu: 是否胡牌
    :return: [],isHu 动作组合牌，是否胡牌
    """
    if isHu:              # 胡牌就直接返回
        return [], True

    # 更新全局变量
    MJ.KING = king_card
    game_config.TIME_START = time.time()
    game_config.LEFT_NUM, discards_list = Utils.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    game_config.LEFT_NUM[Utils.translate16_33(Utils.pre_king(king_card))] -= 1
    # if remain_num == 0:
    #     remain_num = 1
    REMAIN_NUM = sum(game_config.LEFT_NUM)
    if round > 100:
        game_config.T_SELFMO = []
        RT1 = []
        RT2 = []
        RT3 = []
    else:
        game_config.T_SELFMO = [float(i) / REMAIN_NUM for i in game_config.LEFT_NUM]
        RT1 = []
        RT2 = []
        RT3 = []

    # t1tot3_dict = MJ.t1tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])
    # t2tot3_dict = MJ.t2tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])

    # 计算操作前评估值 ？那这个时候手牌是13还是14张牌，对于评估值的计算有影响吗，还是说padding补齐了缺的那张牌？
    cards_pre = copy.copy(cards)
    # cards_pre.append(-1) #加入一张0作为下次摸到的牌，并提升一定的概率a
    # 计算各牌型的评估值和本轮的最小向听数，这里为操作前的评分
    score_dict_pre, min_xts_pre = get_score_dict(cards_pre, suits, king_card, fei_king, padding=[-1])
    # xts_pre = min
    if score_dict_pre != {}:
        score_pre = max(score_dict_pre.values())  # 选一个最大的score值作为操作前
    else:
        score_pre = 0

    # 计算操作后的评估值
    # 确定可选动作
    set_cards = list(set(cards))
    if self_turn:  # 自己回合，暗杠或补杠
        for card in set_cards:
            if cards.count(card) == 4:
                return [card, card, card, card], False  # 暗杠必杠
        for suit in suits:
            if suit.count(suit[0]) == 3 and suit[0] in cards:
                return suit + [suit[0]], False  # 补杠必杠

    else:  # 其他玩家回合 #明杠，吃碰
        if cards.count(op_card) == 3:  # 杠
            return [op_card, op_card, op_card, op_card], False

        op_sets = []  # 可操作的集合
        # print(canchi)
        if canchi:
            # 计算可吃组合
            if op_card < 0x30:  # 字牌不能吃,计算可和操作牌组成顺子的搭子
                rm_sets = [[op_card - 2, op_card - 1], [op_card - 1, op_card + 1],
                           [op_card + 1, op_card + 2]]  # 所有可能操作的集合
            else:
                rm_sets = []
            for op_set in rm_sets:  # 若手牌有有效搭子,则把搭子加入可操作集合
                if op_set[0] in cards and op_set[1] in cards:
                    op_sets.append(op_set)
            # 碰
            if cards.count(op_card) >= 2:
                op_sets.append([op_card, op_card])
        else:
            if cards.count(op_card) >= 2:
                op_sets.append([op_card, op_card])

        # print(op_sets)
        score_set = []
        # 依次计算操作后的评估值如何
        for op_set in op_sets:
            cards_ = copy.copy(cards)
            cards_.remove(op_set[0])
            cards_.remove(op_set[1])

            suits_ = MJ.deepcopy(suits)
            suits_.append(sorted(op_set + [op_card]))  # 进行操作,把新组合加入副露
            # 操作完后计算评估值，操作后的评分
            score_dict, _ = get_score_dict(cards=cards_, suits=suits_, king_card=king_card, fei_king=fei_king,
                                           max_xts=min_xts_pre)  # max_xts用于限制操作完后向听数增加的情况
            # max_discard = max(score_dict, key=lambda x: score_dict[x])
            # print ("score_dict",score_dict)
            if score_dict != {}:
                score = max(score_dict.values())
                score_set.append(score)
        if time.time() - game_config.TIME_START > 3:
            pass
            # logger.warning("op time out %s", time.time() - TIME_START)
        # print('op_set:',op_sets)
        # print('score_set:',score_set)
        # print(score_pre)
        if score_set == []:
            return [], False
        else:
            max_score = max(score_set)
            # print max_score, score_pre
            # print('max_score:', max_score, 'score_pre', score_pre)
            if max_score > score_pre * 1.05:
                return sorted(op_sets[score_set.index(max_score)] + [op_card]), False
    return [], False
