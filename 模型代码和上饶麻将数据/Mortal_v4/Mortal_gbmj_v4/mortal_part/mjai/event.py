import json
from typing import List, Dict, Any

from mortal_part.tile import Tile


class Event:
    # 定义活动事件类
    def __init__(self, *args, **kwargs):
        self.event = self.create_event(*args, **kwargs)

    def __repr__(self):
        return f"Event({self.event})"
    # TODO 添加平台游戏开始事件BotStartGame、开局事件BotStartKyoku
    # 摸牌 / 弃牌 / 碰 / 明杠 / 暗杠 / 补杠 / 过 / 和 / 开局 / 流局 / 对打开局/
    class Tsumo:
        def __init__(self, player, tile: Tile):
            self.player = player
            self.tile = tile

        def __repr__(self):
            return f"Tsumo(player={self.player}, tile={self.tile})"

    class Dahai:
        def __init__(self, player, tile: Tile):
            self.player = player
            self.tile = tile

        def __repr__(self):
            return f"Dahai(player={self.player}, tile={self.tile})"

    class Chi:
        def __init__(self, player, target, tile, consumed: List[Tile]):
            self.player = player
            self.target = target
            self.tile = tile
            self.consumed = consumed

        def __repr__(self):
            return f"Chi(player={self.player}, tile={self.tile}, target={self.target}, consumed={self.consumed})"

    class Pon:
        def __init__(self, player, tile: Tile, target, consumed: List[Tile]):
            self.player = player
            self.tile = tile
            self.target = target
            self.consumed = consumed

        def __repr__(self):
            return f"Pon(player={self.player}, tile={self.tile}, target={self.target}, consumed={self.consumed})"

    class MinGang:
        def __init__(self, player, tile: Tile, target, consumed: List[Tile]):
            self.player = player
            self.tile = tile
            self.target = target
            self.consumed = consumed

        def __repr__(self):
            return f"MinGang(player={self.player}, tile={self.tile}, target={self.target}, consumed={self.consumed})"

    class AnGang:
        def __init__(self, player, consumed: List[Tile]):
            self.player = player
            self.consumed = consumed

        def __repr__(self):
            return f"AnGang(player={self.player}, consumed={self.consumed})"

    class BuGang:
        def __init__(self, player, tile: Tile, consumed: List[Tile]):
            self.player = player
            self.tile = tile
            self.consumed = consumed

        def __repr__(self):
            return f"BuGang(player={self.player}, tile={self.tile}, consumed={self.consumed})"

    class Guo:
        def __init__(self, player):
            self.player = player

        def __repr__(self):
            return f"Guo(player={self.player})"

    class Hu:
        def __init__(self, player, tile: Tile, target, fan, deltas, flower_count=None):
            self.player = player
            self.tile = tile
            self.target = target
            self.fan = fan
            self.deltas = deltas
            self.flower_count = flower_count

        def __repr__(self):
            return f"Hu(player={self.player}, tile={self.tile}, target={self.target}, deltas={self.deltas})"

    class StartGame:
        def __init__(self, names, zhuang, wind, seed=None):
            self.names = names
            self.zhuang = zhuang
            self.wind = wind
            self.seed = seed

        def __repr__(self):
            return f"StartGame(names={self.names}, zhaung={self.zhuang}, wind={self.wind}, seed={self.seed})"

    class StartKyoku:
        def __init__(self, wind, kyoku, zhuang, scores, tehais, flower_count, king_card=None):
            self.wind = wind
            self.kyoku = kyoku
            self.zhuang = zhuang
            self.scores = scores
            self.tehais = tehais
            self.flower_count = flower_count
            # 上饶麻将的精牌；旧国标日志没有该字段，保持 None 即可向后兼容。
            self.king_card = king_card

        def __repr__(self):
            return f"StartKyoku(wind={self.wind}, kyoku={self.kyoku}, zhuang={self.zhuang}, scores={self.scores}, tehais={self.tehais}, flower_count={self.flower_count}, king_card={self.king_card})"

    class EndKyoku:
        def __init__(self):
            pass

        def __repr__(self):
            return f"EndKyoku()"

    class EndGame:
        def __init__(self):
            pass

        def __repr__(self):
            return "EndGame()"

    class Ryukyoku:
        def __init__(self):
            pass

        def __repr__(self):
            return f"Ryukyoku()"

    class BotStartGame:
        def __init__(self, seat_wind, round_wind):
            self.seat_wind = seat_wind
            self.round_wind = round_wind

        def __repr__(self):
            return f"BotStartGame(seat_wind={self.seat_wind}, round_wind={self.round_wind})"

    class BotStartKyoku:
        def __init__(self, tehais):
            self.tehais = tehais

        def __repr__(self):
            return f"BotStartKyoku(tehais={self.tehais})"

    class BuHua:
        def __init__(self, player, flower, replacement, auto: bool):
            self.player = player
            self.flower = flower
            self.replacement = replacement
            self.auto = auto

        def __repr__(self):
            return f"BuHua(player={self.player}, flower={self.flower}, replacement={self.replacement}, auto={self.auto})"

    class NoneEvent:
        def __init__(self):
            pass

        def __repr__(self):
            return "NoneEvent()"

    # 由于Python的Enum不支持变体字段，我们使用一个方法来创建事件
    @staticmethod
    def create_event(type, *args, **kwargs):
        if type == 'Tsumo':
            return Event.Tsumo(*args, **kwargs)
        elif type == 'Dahai':
            return Event.Dahai(*args, **kwargs)
        elif type == 'Chi':
            return Event.Chi(*args, **kwargs)
        elif type == 'Pon':
            return Event.Pon(*args, **kwargs)
        elif type == 'MinGang':
            return Event.MinGang(*args, **kwargs)
        elif type == 'AnGang':
            return Event.AnGang(*args, **kwargs)
        elif type == 'BuGang':
            return Event.BuGang(*args, **kwargs)
        elif type == 'Guo':
            return Event.Guo(*args, **kwargs)
        elif type == 'Hu':
            return Event.Hu(*args, **kwargs)
        elif type == 'StartGame':
            return Event.StartGame(*args, **kwargs)
        elif type == 'StartKyoku':
            return Event.StartKyoku(*args, **kwargs)
        elif type == 'EndKyoku':
            return Event.EndKyoku()
        elif type == 'EndGame':
            return Event.EndGame()
        elif type == 'BotStartGame':
            return Event.BotStartGame(*args, **kwargs)
        elif type == 'BotStartKyoku':
            return Event.BotStartKyoku(*args, **kwargs)
        elif type == 'BuHua':
            return Event.BuHua(*args, **kwargs)
        elif type == 'Ryukyoku':
            return Event.Ryukyoku()
        else:
            return Event.NoneEvent()

    def actor(self):
        if hasattr(self.event, 'player'):
            return self.event.player
        else:
            return None

    def get_tile(self):
        if hasattr(self.event, 'tile'):
            return self.event.tile.id
        else:
            raise ValueError("This event does't have tile attribute")

    def is_in_game_announce(self):
        if type(self.event) == Event.Hu:
            return True
        return False

    def augment(self):
        """
        数据增强方法
        @return:
        """

        def swap_tile(tile: Tile):
            tile.augment()

        if type(self.event) == Event.Tsumo or type(self.event) == Event.Dahai:
            swap_tile(self.event.tile)
        elif type(self.event) == Event.Pon:
            swap_tile(self.event.tile)
            for i in range(2):
                swap_tile(self.event.consumed[i])
        elif type(self.event) == Event.MinGang or type(self.event) == Event.BuGang:
            swap_tile(self.event.tile)
            for i in range(2):
                swap_tile(self.event.consumed[i])
        elif type(self.event) == Event.AnGang:
            for i in range(2):
                swap_tile(self.event.consumed[i])
        else:
            pass

    @staticmethod
    def from_str(json_str: str) -> 'Event':
        """
        从JSON字符串解析Event对象，类似于Rust的json::from_str

        Args:
            json_str: JSON字符串

        Returns:
            解析后的Event对象
        """
        data = json.loads(json_str)
        return Event.from_value(data)

    @staticmethod
    def from_value(data: Dict[str, Any]) -> 'Event':
        """
        从已解析的JSON值创建Event对象

        Args:
            data: 解析后的JSON数据，可以是嵌套格式或扁平格式

        Returns:
            Event对象
        """
        # 检查是否使用扁平格式 {"type": "EventType", ...}
        if "type" in data:
            event_type = data["type"]
            return Event._from_flat_json(event_type, data)

        else:
            raise ValueError("无法识别的JSON格式")

    @staticmethod
    def _from_flat_json(event_type: str, data: Dict[str, Any]) -> 'Event':
        """从扁平格式的JSON创建Event对象"""
        if event_type == 'Tsumo':
            tile = Tile.from_str(data['tile'])
            return Event('Tsumo', data['player'], tile)

        elif event_type == 'Dahai':
            tile = Tile.from_str(data['tile'])
            return Event('Dahai', data['player'], tile)

        elif event_type == 'Chi':
            tile = Tile.from_str(data['tile'])
            consumed = [Tile.from_str(t) for t in data['consumed']]
            return Event('Chi', data['player'], data['target'], tile, consumed)

        elif event_type == 'Pon':
            tile = Tile.from_str(data['tile'])
            consumed = [Tile.from_str(t) for t in data['consumed']]
            return Event('Pon', data['player'], tile, data['target'], consumed)

        elif event_type == 'MinGang':
            tile = Tile.from_str(data['tile'])
            consumed = [Tile.from_str(t) for t in data['consumed']]
            return Event('MinGang', data['player'], tile, data['target'], consumed)

        elif event_type == 'AnGang':
            consumed = [Tile.from_str(t) for t in data['consumed']]
            return Event('AnGang', data['player'], consumed)

        elif event_type == 'BuGang':
            tile = Tile.from_str(data['tile'])
            consumed = [Tile.from_str(t) for t in data['consumed']]
            return Event('BuGang', data['player'], tile, consumed)

        elif event_type == 'Guo':
            return Event('Guo', data['player'])

        elif event_type == 'Hu':
            return Event('Hu', data['player'], Tile.from_str(data['tile']), data['target'], data.get('fan', 0), data.get('deltas', None), data.get('flower_count', [0,0,0,0]))

        elif event_type == 'StartGame':
            return Event('StartGame', data['names'], data.get('zhuang', None), data.get('wind', None))

        elif event_type == 'StartKyoku':
            tehais = [[Tile.from_str(t) for t in player_tiles] for player_tiles in data['tehais']]
            raw_king_card = data.get('king_card')
            king_card = Tile.from_str(raw_king_card) if raw_king_card is not None else None
            return Event('StartKyoku', data.get('wind', None), data['kyoku'], data['zhuang'], data['scores'], tehais, data.get('flower_count', [0, 0, 0, 0]), king_card)

        elif event_type == 'EndKyoku':
            return Event('EndKyoku')

        elif event_type == 'EndGame':
            return Event('EndGame')

        elif event_type == 'Ryukyoku':
            return Event('Ryukyoku')

        elif event_type == 'BuHua':
            return Event('BuHua', data['player'], data['flower'], Tile.from_str(data['replacement']) if not data['replacement'].startswith('H') else data['replacement'], data.get('auto', True))
        else:
            return Event('NoneEvent')

    def to_dict(self) -> Dict[str, Any]:
        """将枚举变体转为可序列化的字典"""
        if isinstance(self.event, Event.StartGame):
            return {
                "type": "StartGame",
                "names": self.event.names,
                "zhuang": self.event.zhuang,
                "seed": self.event.seed
            }
        elif isinstance(self.event, Event.EndGame):
            return {"type": "EndGame"}
        elif isinstance(self.event, Event.StartKyoku):
            result = {
                "type": "StartKyoku",
                "wind": self.event.wind,
                "kyoku": self.event.kyoku,
                "zhuang": self.event.zhuang,
                "scores": self.event.scores,
                "tehais": [[x.__repr__() for x in self.event.tehais[0]], [x.__repr__() for x in self.event.tehais[1]],
                           [x.__repr__() for x in self.event.tehais[2]], [x.__repr__() for x in self.event.tehais[3]]],
                "flower_count": self.event.flower_count,
            }
            if self.event.king_card is not None:
                result["king_card"] = self.event.king_card.__repr__()
            return result
        elif isinstance(self.event, Event.EndKyoku):
            return {
                "type": "EndKyoku",
            }
        elif isinstance(self.event, Event.Ryukyoku):
            return {
                "type": "Ryukyoku",
            }
        elif isinstance(self.event, Event.Tsumo):
            return {
                "type": "Tsumo",
                "player": self.event.player,
                "tile": self.event.tile.__repr__()
            }
        elif isinstance(self.event, Event.Dahai):
            return {
                "type": "Dahai",
                "player": self.event.player,
                "tile": self.event.tile.__repr__()
            }
        elif isinstance(self.event, Event.Chi):
            return {
                "type": "Chi",
                "player": self.event.player,
                "target": self.event.target,
                "tile": self.event.tile.__repr__(),
                "consumed": [t.__repr__() for t in self.event.consumed]
            }
        elif isinstance(self.event, Event.Pon):
            return {
                "type": "Pon",
                "player": self.event.player,
                "target": self.event.target,
                "tile": self.event.tile.__repr__(),
                "consumed": [t.__repr__() for t in self.event.consumed]
            }
        elif isinstance(self.event, Event.Hu):
            return {
                "type": "Hu",
                "player": self.event.player,
                "target": self.event.target,
                "tile": self.event.tile.__repr__(),
                "fan": self.event.fan,
                "deltas": self.event.deltas
            }
        elif isinstance(self.event, Event.MinGang):
            return {
                "type": "MinGang",
                "player": self.event.player,
                "target": self.event.target,
                "tile": self.event.tile.__repr__(),
                "consumed": [t.__repr__() for t in self.event.consumed],
            }
        elif isinstance(self.event, Event.AnGang):
            return {
                "type": "AnGang",
                "player": self.event.player,
                "consumed": [t.__repr__() for t in self.event.consumed],
            }
        elif isinstance(self.event, Event.BuGang):
            return {
                "type": "BuGang",
                "player": self.event.player,
                "tile": self.event.tile.__repr__(),
                "consumed": [t.__repr__() for t in self.event.consumed],
            }
        elif isinstance(self.event, Event.Guo):
            return {
                "type": "Guo",
                "player": self.event.player
            }
        elif isinstance(self.event, Event.BuHua):
            return {
                "type": "BuHua",
                "player": self.event.player,
                "flower": self.event.flower,
                "replacement": self.event.replacement.__repr__() if isinstance(self.event.replacement, Tile) else self.event.replacement,
                "auto": self.event.auto
            }
        else:
            return {"type": "NoneEvent"}


class Metadata:
    """
    定义元数据类，用于存储q值、掩码位、向听数、是否振听等等信息
    """

    def __init__(self, q_values: list[float] = None, mask_bits: int = None, is_greedy: bool = None,
                 batch_size: int = None, eval_time_ns: int = None,
                 shanten: int = None, kan_select: 'Metadata' = None):
        self.q_values = q_values
        self.mask_bits = mask_bits
        self.is_greedy = is_greedy
        self.batch_size = batch_size
        self.eval_time_ns = eval_time_ns
        self.shanten = shanten
        self.kan_select = kan_select

    def to_dict(self):
        return {
            "q_values": self.q_values,
            "mask_bits": self.mask_bits,
            "is_greedy": self.is_greedy,
            "batch_size": self.batch_size,
            "eval_time_ns": self.eval_time_ns,
            "shanten": self.shanten,
        }


class EventExt:
    """
    活动信息类，同时存储牌局中的活动信息和元数据信息
    """

    def __init__(self, event: Event, meta: Metadata):
        self.event = event
        self.meta = meta

    def __repr__(self):
        return f"EventExt(event={self.event}, meta={self.meta})"

    @staticmethod
    def no_meta(event):
        return EventExt(event, None)

    def to_dict(self):
        t = self.event.to_dict()
        if not isinstance(self.event.event, Event.Tsumo):
            t["meta"] = self.meta.to_dict() if self.meta else None
        return t


class EventWithCanAct:
    """
    存储可操作的活动类
    """

    def __init__(self, event: Event, can_act: bool):
        self.event = event
        self.can_act = can_act


class OutOfBoundError(Exception):
    """
    定义一个数值超出表示范围的异常
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        return f"out-of-range number {self.value}"


# 使用示例
# 创建一个摸牌事件
if __name__ == "__main__":
    # t1 = Tile(3)
    # event1 = Event('Tsumo', 0, t1)
    # print(event1.actor())
    # print(type(event1))
    # print(event1.get_tile())
    #
    # t2 = Tile(15)
    # event2 = Event('Dahai', 1, t2, False)
    # print(isinstance(event2.event, Event.Dahai))
    # print(event2.actor())
    # print(type(event2))
    # print(event2.get_tile())
    #
    # m_data = Metadata(0.01, 1, True, 128, 100, 3, False)
    # print(type(m_data))
    #
    # try:
    #     raise OutOfBoundError(255)
    # except OutOfBoundError as e:
    #     print(e)

    event3 = Event('NoneEvent')
    print(isinstance(event3.event, Event.NoneEvent))
    ext = EventExt(event3, None)
    ext2 = EventExt(event3, None)
    t = [ext, ext2]
    print(event3)
    print(t)
