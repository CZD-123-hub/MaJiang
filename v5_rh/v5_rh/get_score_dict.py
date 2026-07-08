import time
from .GameConfig import GameConfig
from .Node_JY import JiuYao
from .Node_PH import PingHu, SearchTree_PH
from .Node_SSL import ShiSanLan
from .Node_Qidui import Qidui
from . import Utils

game_config = GameConfig()

def get_score_dict(cards, suits, king_card, fei_king, padding=[], max_xts=14):
    """
    :param cards: 手牌
    :param suits: 副露
    :param king_card:  宝牌
    :param fei_king: 飞宝
    :param padding: 填充牌。用于计算：op缺一张牌时，填充-1
    :param max_xts: 允许的最大向听数，否则停止计算，用于处理：op中非平胡牌型的吃碰杠处理，例如十三烂牌型吃碰导致需要计算平胡牌型的出牌评估值，从而导致超时
    :return: score_dict,min_xts ，各出牌的评估值与本轮计算的最小向听数（用于op中对比操作前后时）
    """
    # 定义4个牌型类对象
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=fei_king, padding=padding)
    SSL = ShiSanLan(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    JY = JiuYao(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    # 组合信息
    CS_PH = PH.pinghu_CS2()  # [kz,sz,aa,ab,xts,leftCards],向听数最少的平胡组合集合
    CS_SSL = SSL.ssl_CS()  # [[wan],[tiao],[tong],[zi],[left],xts],全部的十三烂组合集合
    CS_JY = JY.yaojiu_CS()  # [[有效牌],[无效牌],xts],全部的九幺组合集合
    CS_QD = QD.qidui_CS()  # [[aa],[单张],xts],就一种七对子组合

    # 向听数
    xts_list = [CS_PH[0][-2], CS_SSL[0][-1], CS_JY[-1], CS_QD[-1]]
    hu_types = ["七小对", "平胡", "十三浪", "九幺"]
    for hu_type,xts in zip(hu_types, xts_list):
        game_config.recommended_hu_types_ting.append({hu_type: xts})

    # -------------------------此处添加监视器类----------------------------------
    # print("xts_list PH,SSL,JY,QD", xts_list)
    # logger.info("xts PH,SSL,JY,QD:%s",xts_list)
    min_xts = min(xts_list)
    # op中吃碰后向听数增加的情况，特别是打非平胡的牌型，此时评估值为0，不推荐打
    if min_xts > max_xts + 1:
        return {cards[-1]: 0}, min_xts

    # 但是向听数最少和次最少的期望值不一定最高，所以这是一个三分类问题，并且问题关建是给数据打标签
    type_list = []  # 需搜索的牌型：最小向听数和最小向听数加1的牌型需要搜索
    for i in range(4):
        if xts_list[i] - 1 <= min_xts:
            type_list.append(i)


    # print("type_list=",type_list)

    score_list = []
    time_start = time.time()
    time_list = []
    for i in type_list:
        if i == 0:
            search_PH = SearchTree_PH(hand=cards, suits=suits, combination_sets=CS_PH, king_card=king_card,fei_king=fei_king)
            score_list.append(search_PH.get_discard_score())
        elif i == 1:
            score_list.append(SSL.get_discard_score())
        elif i == 2:
            score_list.append(JY.get_discard_score())
        elif i == 3:
            score_list.append(QD.get_discard_score())
        time_list.append(time.time() - time_start - sum(time_list))


    # print time_list
    # logger.info("time use%s",time_list)
    # 计算总的评估值，有所有选中的牌型的评估值之和
    score_dict = {}
    # print("score_list", score_list)
    for score in score_list:
        for key in score.keys():
            if key not in score_dict.keys():
                score_dict[key] = score[key] - float(Utils.value_t1(key)) / (10 ** (min_xts + 1) / 2)  # 用来区分相同权重的出牌
            else:
                score_dict[key] += score[key]   # 胡不同的牌型丢弃同一张牌的评估值加起来
    # 飞宝的权重增加
    if king_card in score_dict.keys():
        score_dict[king_card] *= 1.2
    # print("score_list", score_list)
    return score_dict, min_xts
