#include "calsht.hpp"
#include "hash.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
constexpr int NUM_TIDS = 34;
const std::conditional<ENABLE_NYANTEN, NyantenHash<9>, DefaultHash<9>>::type hash1;
const std::conditional<ENABLE_NYANTEN, NyantenHash<7>, DefaultHash<7>>::type hash2;

namespace {

using IntHand34 = std::array<int, NUM_TIDS>;

struct GbmjRouteTableHeader {
  std::array<char, 8> magic{};
  uint32_t version = 1;
  uint32_t fan = 0;
  uint32_t reserved = 0;
  uint32_t record_size = 0;
};

struct MeldDef {
  std::array<int, 3> tiles{};
};

constexpr std::array<char, 8> GBMJ_ROUTE_TABLE_MAGIC{'G', 'B', 'M', 'J', 'R', 'T', '0', '1'};

int suit_of(const int tid)
{
  if (tid < 0 || tid >= 27) return -1;
  return tid / 9;
}

int rank_of(const int tid)
{
  return tid % 9;
}

bool is_honor(const int tid)
{
  return tid >= 27;
}

bool is_wind(const int tid)
{
  return tid >= 27 && tid <= 30;
}

bool is_dragon(const int tid)
{
  return tid >= 31 && tid <= 33;
}

bool is_terminal(const int tid)
{
  return tid < 27 && (rank_of(tid) == 0 || rank_of(tid) == 8);
}

bool is_terminal_or_honor(const int tid)
{
  return is_terminal(tid) || is_honor(tid);
}

std::array<int, 3> chow_tiles(const int suit, const int start)
{
  const int base = suit * 9 + start;
  return {base, base + 1, base + 2};
}

MeldDef triplet_meld(const int tid)
{
  return MeldDef{{tid, tid, tid}};
}

MeldDef chow_meld(const int suit, const int start)
{
  return MeldDef{chow_tiles(suit, start)};
}

std::vector<int> chow_to_tiles(const int suit, const int start)
{
  const auto tiles = chow_tiles(suit, start);
  return {tiles[0], tiles[1], tiles[2]};
}

std::vector<int> concat_tiles(const std::initializer_list<std::vector<int>> parts)
{
  std::vector<int> ret;
  for (const auto& part : parts) {
    ret.insert(ret.end(), part.begin(), part.end());
  }
  return ret;
}

const std::vector<std::vector<int>>& mixed_shifted_chows_cores()
{
  static const std::vector<std::vector<int>> cores = [] {
    std::vector<std::vector<int>> ret;
    std::array<int, 3> suits{0, 1, 2};
    do {
      for (int start = 0; start <= 4; ++start) {
        ret.push_back(concat_tiles({
          chow_to_tiles(suits[0], start),
          chow_to_tiles(suits[1], start + 1),
          chow_to_tiles(suits[2], start + 2),
        }));
      }
    } while (std::next_permutation(suits.begin(), suits.end()));
    return ret;
  }();
  return cores;
}

const std::vector<std::vector<int>>& mixed_triple_chows_cores()
{
  static const std::vector<std::vector<int>> cores = [] {
    std::vector<std::vector<int>> ret;
    for (int start = 0; start <= 6; ++start) {
      ret.push_back(concat_tiles({
        chow_to_tiles(0, start),
        chow_to_tiles(1, start),
        chow_to_tiles(2, start),
      }));
    }
    return ret;
  }();
  return cores;
}

const std::vector<std::vector<int>>& mixed_straight_cores()
{
  static const std::vector<std::vector<int>> cores = [] {
    std::vector<std::vector<int>> ret;
    std::array<int, 3> suits{0, 1, 2};
    do {
      ret.push_back(concat_tiles({
        chow_to_tiles(suits[0], 0),
        chow_to_tiles(suits[1], 3),
        chow_to_tiles(suits[2], 6),
      }));
    } while (std::next_permutation(suits.begin(), suits.end()));
    return ret;
  }();
  return cores;
}

const std::vector<std::vector<int>>& pure_straight_cores()
{
  static const std::vector<std::vector<int>> cores = [] {
    std::vector<std::vector<int>> ret;
    for (int suit = 0; suit < 3; ++suit) {
      ret.push_back(concat_tiles({
        chow_to_tiles(suit, 0),
        chow_to_tiles(suit, 3),
        chow_to_tiles(suit, 6),
      }));
    }
    return ret;
  }();
  return cores;
}

const std::vector<std::vector<int>>& pure_shifted_chows_cores()
{
  static const std::vector<std::vector<int>> cores = [] {
    std::vector<std::vector<int>> ret;
    for (int suit = 0; suit < 3; ++suit) {
      for (int start = 0; start <= 4; ++start) {
        ret.push_back(concat_tiles({
          chow_to_tiles(suit, start),
          chow_to_tiles(suit, start + 1),
          chow_to_tiles(suit, start + 2),
        }));
      }
      for (int start = 0; start <= 2; ++start) {
        ret.push_back(concat_tiles({
          chow_to_tiles(suit, start),
          chow_to_tiles(suit, start + 2),
          chow_to_tiles(suit, start + 4),
        }));
      }
    }
    return ret;
  }();
  return cores;
}

const std::vector<std::vector<int>>& knitted_straight_cores()
{
  static const std::vector<std::vector<int>> cores = [] {
    std::vector<std::vector<int>> ret;
    std::array<int, 3> suits{0, 1, 2};
    const std::array<std::array<int, 3>, 3> rank_groups{{
      {0, 3, 6},
      {1, 4, 7},
      {2, 5, 8},
    }};

    do {
      std::vector<int> required;
      required.reserve(9);
      for (int group = 0; group < 3; ++group) {
        for (const int rank : rank_groups[group]) {
          required.push_back(suits[group] * 9 + rank);
        }
      }
      ret.push_back(required);
    } while (std::next_permutation(suits.begin(), suits.end()));
    return ret;
  }();
  return cores;
}

std::filesystem::path gbmj_table_path(const std::filesystem::path& dir,
                                      const Calsht::GbmjFan fan)
{
  const auto& names = Calsht::gbmj_fan_names();
  return dir / ("gbmj_" + std::string(names[static_cast<int>(fan)]) + ".bin");
}

int inferred_meld_count(const IntHand34& hand)
{
  const int n = std::accumulate(hand.begin(), hand.end(), 0);
  return std::min(4, std::max(0, n / 3));
}

IntHand34 filtered_hand(const IntHand34& hand, const std::function<bool(int)>& keep)
{
  IntHand34 ret{};
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    if (keep(tid)) ret[tid] = hand[tid];
  }
  return ret;
}

int regular_distance_in_mask(const Calsht& calsht,
                             const IntHand34& hand,
                             const std::function<bool(int)>& keep)
{
    // [GBMJ main-fan shanten] Reuse the original shanten-number dense
    // suit/honor tables instead of scanning complete-hand route lists.  The
    // regular form is still a 4-meld-1-pair hand, but only tiles accepted by
    // keep() are allowed to contribute.
  return calsht.calc_lh(filtered_hand(hand, keep), 4);
}

int full_flush_distance(const Calsht& calsht, const IntHand34& hand)
{
  int best = 15;
  for (int suit = 0; suit < 3; ++suit) {
    best = std::min(best, regular_distance_in_mask(
                              calsht,
                              hand,
                              [suit](const int tid) { return suit_of(tid) == suit; }));
  }
  return best;
}

int greater_than_five_distance(const Calsht& calsht, const IntHand34& hand)
{
  return regular_distance_in_mask(calsht, hand, [](const int tid) {
    return tid < 27 && rank_of(tid) >= 5;
  });
}

int less_than_five_distance(const Calsht& calsht, const IntHand34& hand)
{
  return regular_distance_in_mask(calsht, hand, [](const int tid) {
    return tid < 27 && rank_of(tid) <= 3;
  });
}

int all_pungs_distance(const IntHand34& hand)
{
    // [GBMJ main-fan shanten] Exact direct formula for Peng Peng Hu:
    // choose one pair tile and four different triplet tiles.  This keeps the
    // same distance convention used by the original table code
    // (complete = 0, one tile missing = 1).
  int best = 15;

  for (int pair_tid = 0; pair_tid < NUM_TIDS; ++pair_tid) {
    const int pair_missing = std::max(0, 2 - hand[pair_tid]);
    std::array<int, NUM_TIDS - 1> triplet_missing{};
    int n = 0;

    for (int tid = 0; tid < NUM_TIDS; ++tid) {
      if (tid == pair_tid) continue;
      triplet_missing[n++] = std::max(0, 3 - hand[tid]);
    }

    std::nth_element(triplet_missing.begin(),
                     triplet_missing.begin() + 4,
                     triplet_missing.end());
    int missing = pair_missing;
    for (int i = 0; i < 4; ++i) missing += triplet_missing[i];
    best = std::min(best, missing);
  }

  return best;
}

int missing_all_types(const IntHand34& hand)
{
  int missing = 0;
  auto has_any = [&hand](const int first, const int last_exclusive) {
    for (int tid = first; tid < last_exclusive; ++tid) {
      if (hand[tid] > 0) return true;
    }
    return false;
  };

  if (!has_any(0, 9)) ++missing;
  if (!has_any(9, 18)) ++missing;
  if (!has_any(18, 27)) ++missing;
  if (!has_any(27, 31)) ++missing;
  if (!has_any(31, 34)) ++missing;
  return missing;
}

int half_flush_distance(const Calsht& calsht, const IntHand34& hand)
{
  int best = 15;
  const int need_honor = std::any_of(hand.begin() + 27, hand.end(),
                                    [](const int n) { return n > 0; }) ? 0 : 1;
  for (int suit = 0; suit < 3; ++suit) {
    const int dist = regular_distance_in_mask(
        calsht,
        hand,
        [suit](const int tid) { return tid >= 27 || suit_of(tid) == suit; });
    best = std::min(best, std::max(dist, need_honor));
  }
  return best;
}

int all_unrelated_distance(const IntHand34& hand)
{
  int best = 15;
  std::array<int, 3> suits{0, 1, 2};
  const std::array<std::array<int, 3>, 3> rank_groups{{
    {0, 3, 6},
    {1, 4, 7},
    {2, 5, 8},
  }};

  do {
    int have = 0;
    for (int group = 0; group < 3; ++group) {
      for (const int rank : rank_groups[group]) {
        have += hand[suits[group] * 9 + rank] > 0 ? 1 : 0;
      }
    }
    for (int tid = 27; tid < 34; ++tid) {
      have += hand[tid] > 0 ? 1 : 0;
    }
    best = std::min(best, 14 - have);
  } while (std::next_permutation(suits.begin(), suits.end()));

  return best;
}

int meld_missing(const IntHand34& hand, const MeldDef& meld)
{
  IntHand34 need{};
  for (const int tid : meld.tiles) ++need[tid];

  int missing = 0;
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    missing += std::max(0, need[tid] - hand[tid]);
  }
  return missing;
}

int pair_missing(const IntHand34& hand, const int tid)
{
  return std::max(0, 2 - hand[tid]);
}

int outside_hand_distance(const IntHand34& hand)
{
    // [GBMJ route-table rewrite] Exact route-style evaluator for Outside Hand:
    // allowed melds are terminal/honor triplets plus 123/789 chows, and the pair
    // must be terminal/honor.
  std::vector<MeldDef> melds;
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    if (is_terminal_or_honor(tid)) melds.push_back(triplet_meld(tid));
  }
  for (int suit = 0; suit < 3; ++suit) {
    melds.push_back(chow_meld(suit, 0));
    melds.push_back(chow_meld(suit, 6));
  }

  std::vector<int> pair_tiles;
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    if (is_terminal_or_honor(tid)) pair_tiles.push_back(tid);
  }

  int best = 15;
  std::array<int, 4> chosen{};
  std::function<void(int, int)> dfs = [&](const int depth, const int min_mid) {
    if (depth == 4) {
      IntHand34 used{};
      for (const int mid : chosen) {
        for (const int tid : melds[mid].tiles) ++used[tid];
      }

      int meld_dist = 0;
      for (int tid = 0; tid < NUM_TIDS; ++tid) {
        meld_dist += std::max(0, used[tid] - hand[tid]);
        if (meld_dist >= best) return;
      }

      for (const int pair_tid : pair_tiles) {
        IntHand34 need = used;
        need[pair_tid] += 2;
        bool valid = true;
        int dist = 0;
        for (int tid = 0; tid < NUM_TIDS; ++tid) {
          if (need[tid] > 4) {
            valid = false;
            break;
          }
          dist += std::max(0, need[tid] - hand[tid]);
          if (dist >= best) break;
        }
        if (valid) best = std::min(best, dist);
      }
      return;
    }

    for (int mid = min_mid; mid < static_cast<int>(melds.size()); ++mid) {
      chosen[depth] = mid;
      dfs(depth + 1, mid);
    }
  };

  dfs(0, 0);
  return best;
}

bool valid_fixed_meld_shape(const Calsht::GbmjMeld& meld)
{
  if (meld.tile_count != 3 && meld.tile_count != 4) return false;

  std::vector<int> tiles;
  tiles.reserve(static_cast<std::size_t>(meld.tile_count));
  for (int i = 0; i < meld.tile_count; ++i) {
    const int tid = meld.tiles[i];
    if (tid < 0 || tid >= NUM_TIDS) return false;
    tiles.push_back(tid);
  }
  std::sort(tiles.begin(), tiles.end());

  const bool same = std::all_of(tiles.begin(), tiles.end(), [&](const int tid) {
    return tid == tiles.front();
  });
  if (same) return true;

  if (meld.tile_count != 3) return false;
  if (is_honor(tiles[0]) || suit_of(tiles[0]) != suit_of(tiles[1]) ||
      suit_of(tiles[1]) != suit_of(tiles[2])) {
    return false;
  }
  return rank_of(tiles[0]) + 1 == rank_of(tiles[1]) &&
         rank_of(tiles[1]) + 1 == rank_of(tiles[2]);
}

bool fixed_meld_is_triplet_like(const Calsht::GbmjMeld& meld)
{
  if (meld.tile_count != 3 && meld.tile_count != 4) return false;
  for (int i = 1; i < meld.tile_count; ++i) {
    if (meld.tiles[i] != meld.tiles[0]) return false;
  }
  return meld.tiles[0] >= 0 && meld.tiles[0] < NUM_TIDS;
}

bool fixed_meld_equals(const Calsht::GbmjMeld& fixed, const MeldDef& route_meld)
{
  if (fixed.tile_count != 3) return false;

  std::array<int, 3> lhs{fixed.tiles[0], fixed.tiles[1], fixed.tiles[2]};
  auto rhs = route_meld.tiles;
  std::sort(lhs.begin(), lhs.end());
  std::sort(rhs.begin(), rhs.end());
  return lhs == rhs;
}

bool fixed_meld_is_outside(const Calsht::GbmjMeld& meld)
{
  if (!valid_fixed_meld_shape(meld)) return false;
  if (fixed_meld_is_triplet_like(meld)) {
    return is_terminal_or_honor(meld.tiles[0]);
  }

  std::array<int, 3> tiles{meld.tiles[0], meld.tiles[1], meld.tiles[2]};
  std::sort(tiles.begin(), tiles.end());
  return !is_honor(tiles[0]) && suit_of(tiles[0]) == suit_of(tiles[2]) &&
         ((rank_of(tiles[0]) == 0 && rank_of(tiles[2]) == 2) ||
          (rank_of(tiles[0]) == 6 && rank_of(tiles[2]) == 8));
}

bool add_fixed_counts(const std::vector<Calsht::GbmjMeld>& fixed_melds,
                      IntHand34& fixed_counts)
{
  fixed_counts.fill(0);
  for (const auto& meld : fixed_melds) {
    if (!valid_fixed_meld_shape(meld)) return false;
    for (int i = 0; i < meld.tile_count; ++i) {
      const int tid = meld.tiles[i];
      if (++fixed_counts[tid] > 4) return false;
    }
  }
  return true;
}

IntHand34 merge_counts(const IntHand34& concealed, const IntHand34& fixed_counts)
{
  IntHand34 owned = concealed;
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    owned[tid] += fixed_counts[tid];
  }
  return owned;
}

int fixed_meld_slots(const std::vector<Calsht::GbmjMeld>& fixed_melds)
{
  return static_cast<int>(fixed_melds.size());
}

bool fixed_tiles_satisfy(const std::vector<Calsht::GbmjMeld>& fixed_melds,
                         const std::function<bool(int)>& keep)
{
  for (const auto& meld : fixed_melds) {
    if (!valid_fixed_meld_shape(meld)) return false;
    for (int i = 0; i < meld.tile_count; ++i) {
      if (!keep(meld.tiles[i])) return false;
    }
  }
  return true;
}

std::vector<std::array<MeldDef, 3>> mixed_shifted_chows_meld_cores()
{
  std::vector<std::array<MeldDef, 3>> ret;
  std::array<int, 3> suits{0, 1, 2};
  do {
    for (int start = 0; start <= 4; ++start) {
      ret.push_back({chow_meld(suits[0], start),
                     chow_meld(suits[1], start + 1),
                     chow_meld(suits[2], start + 2)});
    }
  } while (std::next_permutation(suits.begin(), suits.end()));
  return ret;
}

std::vector<std::array<MeldDef, 3>> mixed_triple_chows_meld_cores()
{
  std::vector<std::array<MeldDef, 3>> ret;
  for (int start = 0; start <= 6; ++start) {
    ret.push_back({chow_meld(0, start), chow_meld(1, start), chow_meld(2, start)});
  }
  return ret;
}

std::vector<std::array<MeldDef, 3>> mixed_straight_meld_cores()
{
  std::vector<std::array<MeldDef, 3>> ret;
  std::array<int, 3> suits{0, 1, 2};
  do {
    ret.push_back({chow_meld(suits[0], 0), chow_meld(suits[1], 3), chow_meld(suits[2], 6)});
  } while (std::next_permutation(suits.begin(), suits.end()));
  return ret;
}

std::vector<std::array<MeldDef, 3>> pure_straight_meld_cores()
{
  std::vector<std::array<MeldDef, 3>> ret;
  for (int suit = 0; suit < 3; ++suit) {
    ret.push_back({chow_meld(suit, 0), chow_meld(suit, 3), chow_meld(suit, 6)});
  }
  return ret;
}

std::vector<std::array<MeldDef, 3>> pure_shifted_chows_meld_cores()
{
  std::vector<std::array<MeldDef, 3>> ret;
  for (int suit = 0; suit < 3; ++suit) {
    for (int start = 0; start <= 4; ++start) {
      ret.push_back({chow_meld(suit, start),
                     chow_meld(suit, start + 1),
                     chow_meld(suit, start + 2)});
    }
    for (int start = 0; start <= 2; ++start) {
      ret.push_back({chow_meld(suit, start),
                     chow_meld(suit, start + 2),
                     chow_meld(suit, start + 4)});
    }
  }
  return ret;
}

int route_meld_core_distance_with_fixed(
    const Calsht& calsht,
    const IntHand34& concealed,
    const std::vector<Calsht::GbmjMeld>& fixed_melds,
    const std::vector<std::array<MeldDef, 3>>& cores)
{
  const int fixed_count = fixed_meld_slots(fixed_melds);
  if (fixed_count > 4) return 15;

  int best = 15;
  for (const auto& core : cores) {
    std::array<bool, 3> used_core{};
    int extra_fixed_melds = 0;

    for (const auto& fixed : fixed_melds) {
      bool matched = false;
      for (int i = 0; i < 3; ++i) {
        if (!used_core[i] && fixed_meld_equals(fixed, core[i])) {
          used_core[i] = true;
          matched = true;
          break;
        }
      }
      if (!matched) ++extra_fixed_melds;
    }

    const int free_melds = 1 - extra_fixed_melds;
    if (free_melds < 0) continue;

    IntHand34 residual = concealed;
    int missing = 0;
    for (int i = 0; i < 3; ++i) {
      if (used_core[i]) continue;
      for (const int tid : core[i].tiles) {
        if (residual[tid] > 0) {
          --residual[tid];
        }
        else {
          ++missing;
        }
      }
    }

    best = std::min(best, missing + calsht.calc_lh(residual, free_melds));
  }
  return best;
}

int knitted_straight_distance_with_fixed(const Calsht& calsht,
                                         const IntHand34& concealed,
                                         const std::vector<Calsht::GbmjMeld>& fixed_melds)
{
  const int fixed_count = fixed_meld_slots(fixed_melds);
  if (fixed_count > 1) return 15;

  int best = 15;
  for (const auto& core : knitted_straight_cores()) {
    IntHand34 residual = concealed;
    int missing = 0;
    for (const int tid : core) {
      if (residual[tid] > 0) {
        --residual[tid];
      }
      else {
        ++missing;
      }
    }
    best = std::min(best, missing + calsht.calc_lh(residual, 1 - fixed_count));
  }
  return best;
}

int regular_distance_with_fixed(const Calsht& calsht,
                                const IntHand34& concealed,
                                const int fixed_count)
{
  if (fixed_count > 4) return 15;
  return calsht.calc_lh(concealed, 4 - fixed_count);
}

int regular_masked_distance_with_fixed(const Calsht& calsht,
                                       const IntHand34& concealed,
                                       const int fixed_count,
                                       const std::function<bool(int)>& keep)
{
  if (fixed_count > 4) return 15;
  return calsht.calc_lh(filtered_hand(concealed, keep), 4 - fixed_count);
}

int full_flush_distance_with_fixed(const Calsht& calsht,
                                   const IntHand34& concealed,
                                   const std::vector<Calsht::GbmjMeld>& fixed_melds)
{
  int best = 15;
  for (int suit = 0; suit < 3; ++suit) {
    auto keep = [suit](const int tid) { return suit_of(tid) == suit; };
    if (!fixed_tiles_satisfy(fixed_melds, keep)) continue;
    best = std::min(best, regular_masked_distance_with_fixed(
                              calsht, concealed, fixed_meld_slots(fixed_melds), keep));
  }
  return best;
}

int half_flush_distance_with_fixed(const Calsht& calsht,
                                   const IntHand34& concealed,
                                   const IntHand34& owned,
                                   const std::vector<Calsht::GbmjMeld>& fixed_melds)
{
  int best = 15;
  const int need_honor = std::any_of(owned.begin() + 27, owned.end(),
                                    [](const int n) { return n > 0; }) ? 0 : 1;
  for (int suit = 0; suit < 3; ++suit) {
    auto keep = [suit](const int tid) { return tid >= 27 || suit_of(tid) == suit; };
    if (!fixed_tiles_satisfy(fixed_melds, keep)) continue;
    const int dist = regular_masked_distance_with_fixed(
        calsht, concealed, fixed_meld_slots(fixed_melds), keep);
    best = std::min(best, std::max(dist, need_honor));
  }
  return best;
}

int all_pungs_distance_with_fixed(const IntHand34& concealed,
                                  const IntHand34& fixed_counts,
                                  const IntHand34& owned,
                                  const int fixed_count)
{
  if (fixed_count > 4) return 15;
  const int need_triplets = 4 - fixed_count;
  int best = 15;

  for (int pair_tid = 0; pair_tid < NUM_TIDS; ++pair_tid) {
    if (fixed_counts[pair_tid] + 2 > 4) continue;
    const int pair_missing = std::max(0, 2 - concealed[pair_tid]);

    std::array<int, NUM_TIDS> triplet_missing{};
    int n = 0;
    for (int tid = 0; tid < NUM_TIDS; ++tid) {
      if (tid == pair_tid || fixed_counts[tid] > 0) continue;
      triplet_missing[n++] = std::max(0, 3 - owned[tid]);
    }
    if (n < need_triplets) continue;

    std::nth_element(triplet_missing.begin(),
                     triplet_missing.begin() + need_triplets,
                     triplet_missing.begin() + n);
    int missing = pair_missing;
    for (int i = 0; i < need_triplets; ++i) missing += triplet_missing[i];
    best = std::min(best, missing);
  }
  return best;
}

int outside_hand_distance_with_fixed(const IntHand34& fixed_counts,
                                     const IntHand34& owned,
                                     const std::vector<Calsht::GbmjMeld>& fixed_melds)
{
  for (const auto& meld : fixed_melds) {
    if (!fixed_meld_is_outside(meld)) return 15;
  }

  const int need_melds = 4 - fixed_meld_slots(fixed_melds);
  if (need_melds < 0) return 15;

  std::vector<MeldDef> melds;
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    if (is_terminal_or_honor(tid)) melds.push_back(triplet_meld(tid));
  }
  for (int suit = 0; suit < 3; ++suit) {
    melds.push_back(chow_meld(suit, 0));
    melds.push_back(chow_meld(suit, 6));
  }

  std::vector<int> pair_tiles;
  for (int tid = 0; tid < NUM_TIDS; ++tid) {
    if (is_terminal_or_honor(tid)) pair_tiles.push_back(tid);
  }

  int best = 15;
  std::vector<int> chosen(static_cast<std::size_t>(need_melds));
  std::function<void(int, int)> dfs = [&](const int depth, const int min_mid) {
    if (depth == need_melds) {
      IntHand34 base = fixed_counts;
      for (const int mid : chosen) {
        for (const int tid : melds[mid].tiles) ++base[tid];
      }

      for (const int pair_tid : pair_tiles) {
        IntHand34 need = base;
        need[pair_tid] += 2;
        int dist = 0;
        bool valid = true;
        for (int tid = 0; tid < NUM_TIDS; ++tid) {
          if (need[tid] > 4) {
            valid = false;
            break;
          }
          dist += std::max(0, need[tid] - owned[tid]);
          if (dist >= best) break;
        }
        if (valid) best = std::min(best, dist);
      }
      return;
    }

    for (int mid = min_mid; mid < static_cast<int>(melds.size()); ++mid) {
      chosen[static_cast<std::size_t>(depth)] = mid;
      dfs(depth + 1, mid);
    }
  };

  dfs(0, 0);
  return best;
}

} // namespace

Calsht::RVec Calsht::index1(const int n) const
{
  RVec ret(10, 14u);

  ret[0] = 0u;
  ret[1] = std::max(3u - n, 0u);
  ret[5] = std::max(2u - n, 0u);

  return ret;
}

void Calsht::add1(LVec& lhs, const RVec& rhs, const int m) const
{
  for (int j = m + 5; j >= 5; --j) {
    int sht = std::min(lhs[j] + rhs[0], lhs[0] + rhs[j]);

    for (int k = 5; k < j; ++k) {
      sht = std::min({sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k]});
    }

    lhs[j] = sht;
  }

  for (int j = m; j >= 0; --j) {
    int sht = lhs[j] + rhs[0];

    for (int k = 0; k < j; ++k) {
      sht = std::min(sht, lhs[k] + rhs[j - k]);
    }

    lhs[j] = sht;
  }
}

void Calsht::add2(LVec& lhs, const RVec& rhs, const int m) const
{
  const int j = m + 5;
  int sht = std::min(lhs[j] + rhs[0], lhs[0] + rhs[j]);

  for (int k = 5; k < j; ++k) {
    sht = std::min({sht, lhs[k] + rhs[j - k], lhs[j - k] + rhs[k]});
  }

  lhs[j] = sht;
}

void Calsht::read_file(Iter first, Iter last, std::filesystem::path file) const
{
  std::ifstream fin(file, std::ios_base::in | std::ios_base::binary);

  if (!fin) {
    throw std::runtime_error("Reading file does not exist: " + file.string());
  }

  for (; first != last; ++first) {
    fin.read(reinterpret_cast<char*>(first->data()), first->size() * sizeof(RVec::value_type));
  }
}

int Calsht::calc_one_meld_pair_distance(const std::array<int, NUM_TIDS>& t) const
{
  if (!standard_tables_ready_) {
    throw std::logic_error("standard shanten tables are not initialized");
  }

  // [GBMJ key/value route table] The original shanten-number table already is
  // a dense key-value cache:
  //   9-tile suit key -> distances for 0..4 melds and 0..4 melds + pair
  //   7-tile honor key -> same layout without chows
  // Index 1 is "one meld", index 5 is "one pair", index 6 is
  // "one meld plus one pair".  Combining the four independent regions gives
  // the exact minimum distance for the remaining 5 tiles after a 9-tile fan
  // core has been reserved.
  const std::array<const RVec*, 4> parts{
    &mp1[hash1(t.cbegin())],
    &mp1[hash1(t.cbegin() + 9)],
    &mp1[hash1(t.cbegin() + 18)],
    &mp2[hash2(t.cbegin() + 27)],
  };

  int best = 15;
  for (const RVec* part : parts) {
    best = std::min(best, static_cast<int>((*part)[6]));
  }

  for (int meld_part = 0; meld_part < 4; ++meld_part) {
    for (int pair_part = 0; pair_part < 4; ++pair_part) {
      if (meld_part == pair_part) continue;
      best = std::min(best,
                      static_cast<int>((*parts[meld_part])[1]) +
                          static_cast<int>((*parts[pair_part])[5]));
    }
  }

  return best;
}

int Calsht::calc_required_core_distance(const std::array<int, NUM_TIDS>& t,
                                        const std::vector<int>& required) const
{
  std::array<int, NUM_TIDS> residual = t;
  int missing = 0;

  for (const int tid : required) {
    if (residual[tid] > 0) {
      --residual[tid];
    }
    else {
      ++missing;
    }
  }

  return missing + calc_one_meld_pair_distance(residual);
}

int Calsht::calc_template_core_distance(const std::array<int, NUM_TIDS>& t,
                                        const std::vector<std::vector<int>>& cores) const
{
  int best = 15;
  for (const auto& core : cores) {
    best = std::min(best, calc_required_core_distance(t, core));
    if (best == 0) break;
  }
  return best;
}

void Calsht::initialize(const std::string& dir)
{
  read_file(mp1.begin(), mp1.end(), std::filesystem::path(dir) / "shanten_shu.bin");
  read_file(mp2.begin(), mp2.end(), std::filesystem::path(dir) / "shanten_zi.bin");
  standard_tables_ready_ = true;
}

void Calsht::initialize_gbmj_tables(const std::string& dir)
{
  const std::filesystem::path root(dir);

  // [GBMJ key/value route table] If the original dense shanten tables are
    // placed in the same directory, load them automatically.  Several GBMJ
    // route distances then become true key-value lookups instead of
    // complete-hand route-list scans.
  const auto shu_path = root / "shanten_shu.bin";
  const auto zi_path = root / "shanten_zi.bin";
  if (std::filesystem::is_regular_file(shu_path) &&
      std::filesystem::is_regular_file(zi_path)) {
    read_file(mp1.begin(), mp1.end(), shu_path);
    read_file(mp2.begin(), mp2.end(), zi_path);
    standard_tables_ready_ = true;
  }
}

void Calsht::make_gbmj_tables(const std::string& dir)
{
  // [GBMJ main-fan route metadata] Dump one tiny file per main fan so the
  // table directory records which fan routes are supported.  The heavy
  // key-value payload remains the original shanten_shu.bin/shanten_zi.bin
  // segment tables; we deliberately do not dump complete-hand route lists.
  const std::filesystem::path root(dir);
  std::filesystem::create_directories(root);

  for (int i = 0; i < NUM_GBMJ_MAIN_FANS; ++i) {
    const auto fan = static_cast<GbmjFan>(i);
    const auto path = gbmj_table_path(root, fan);
    std::ofstream fout(path, std::ios_base::out | std::ios_base::binary);
    if (!fout) {
      throw std::runtime_error("Failed to open GBMJ route table: " + path.string());
    }

    GbmjRouteTableHeader header;
    header.magic = GBMJ_ROUTE_TABLE_MAGIC;
    header.fan = static_cast<uint32_t>(i);
    header.reserved = 0u;
    fout.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!fout) {
      throw std::runtime_error("Failed to write GBMJ route table: " + path.string());
    }
  }
}

int Calsht::calc_lh(const std::array<int, NUM_TIDS>& t, const int m, const bool three_player) const
{
  LVec ret = mp2[hash2(t.cbegin() + 27)];

  add1(ret, mp1[hash1(t.cbegin() + 18)], m);
  add1(ret, mp1[hash1(t.cbegin() + 9)], m);

  if (three_player) {
    add1(ret, index1(t[8]), m);
    add2(ret, index1(t[0]), m);
  }
  else {
    add2(ret, mp1[hash1(t.cbegin())], m);
  }

  return ret[m + 5];
}

int Calsht::calc_sp(const std::array<int, NUM_TIDS>& t, const bool three_player) const
{
  int pair = 0;
  int kind = 0;

  for (int i = 0; i < NUM_TIDS; ++i) {
    if (three_player && i > 0 && i < 8) continue;

    if (t[i] > 0) {
      ++kind;
      if (t[i] >= 2) ++pair;
    }
  }

  return 7 - pair + (kind < 7 ? 7 - kind : 0);
}

int Calsht::calc_to(const std::array<int, NUM_TIDS>& t) const
{
  int pair = 0;
  int kind = 0;

  for (const int i : {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}) {
    if (t[i] > 0) {
      ++kind;
      if (t[i] >= 2) ++pair;
    }
  }

  return 14 - kind - (pair > 0 ? 1 : 0);
}

int Calsht::calc_gbmj_fan(const std::array<int, NUM_TIDS>& t,
                          const GbmjFan fan,
                          const bool three_player) const
{
  // [GBMJ main-fan shanten] Seven pairs can be computed exactly with the
  // library's original formula.  NoPointHand and FullyMeldedHand are not pure
  // concealed-tile shape fans, so they intentionally fall back to regular-form
  // distance and should be weighted cautiously by the RL potential function.
  if (fan == GbmjFan::SevenPairs) {
    return calc_sp(t, three_player);
  }

  if (fan == GbmjFan::AllPungs) {
    return all_pungs_distance(t);
  }

  if (fan == GbmjFan::AllUnrelated) {
    return all_unrelated_distance(t);
  }

  if (fan == GbmjFan::OutsideHand) {
    return outside_hand_distance(t);
  }

  if (standard_tables_ready_) {
    if (fan == GbmjFan::MixedShiftedChows) {
      return calc_template_core_distance(t, mixed_shifted_chows_cores());
    }
    if (fan == GbmjFan::MixedTripleChows) {
      return calc_template_core_distance(t, mixed_triple_chows_cores());
    }
    if (fan == GbmjFan::MixedStraight) {
      return calc_template_core_distance(t, mixed_straight_cores());
    }
    if (fan == GbmjFan::PureStraight) {
      return calc_template_core_distance(t, pure_straight_cores());
    }
    if (fan == GbmjFan::PureShiftedChows) {
      return calc_template_core_distance(t, pure_shifted_chows_cores());
    }
    if (fan == GbmjFan::KnittedStraight) {
      return calc_template_core_distance(t, knitted_straight_cores());
    }
    if (fan == GbmjFan::FullFlush) {
      return full_flush_distance(*this, t);
    }
    if (fan == GbmjFan::HalfFlush) {
      return half_flush_distance(*this, t);
    }
    if (fan == GbmjFan::AllTypes) {
      return std::max(calc_lh(t, 4, three_player), missing_all_types(t));
    }
    if (fan == GbmjFan::GreaterThanFive) {
      return greater_than_five_distance(*this, t);
    }
    if (fan == GbmjFan::LessThanFive) {
      return less_than_five_distance(*this, t);
    }
  }

  if (fan == GbmjFan::NoPointHand || fan == GbmjFan::FullyMeldedHand) {
    return calc_lh(t, inferred_meld_count(t), three_player);
  }

  throw std::logic_error(
      "GBMJ fan route requires shanten_shu.bin/shanten_zi.bin; call initialize() or place them in the GBMJ table dir");
}

int Calsht::calc_gbmj_fan(const std::array<int, NUM_TIDS>& concealed,
                          const std::vector<GbmjMeld>& fixed_melds,
                          const GbmjFan fan,
                          const bool three_player) const
{
  // [GBMJ fixed-meld route] Open melds are locked structures.  They are not
  // merged and then freely regrouped; every fan route first checks whether the
  // fixed melds can legally belong to that route, then only the concealed
  // remainder is searched/queried.
  if (fixed_melds.empty()) {
    return calc_gbmj_fan(concealed, fan, three_player);
  }

  IntHand34 fixed_counts{};
  if (!add_fixed_counts(fixed_melds, fixed_counts)) return 15;
  const IntHand34 owned = merge_counts(concealed, fixed_counts);
  const int fixed_count = fixed_meld_slots(fixed_melds);
  if (fixed_count > 4) return 15;

  if (fan == GbmjFan::SevenPairs || fan == GbmjFan::AllUnrelated) {
    return 15;
  }

  if (fan == GbmjFan::AllPungs) {
    for (const auto& meld : fixed_melds) {
      if (!fixed_meld_is_triplet_like(meld)) return 15;
    }
    return all_pungs_distance_with_fixed(concealed, fixed_counts, owned, fixed_count);
  }

  if (fan == GbmjFan::OutsideHand) {
    return outside_hand_distance_with_fixed(fixed_counts, owned, fixed_melds);
  }

  if (!standard_tables_ready_) {
    throw std::logic_error(
        "GBMJ fixed-meld route requires shanten_shu.bin/shanten_zi.bin; call initialize() or place them in the GBMJ table dir");
  }

  if (fan == GbmjFan::MixedShiftedChows) {
    return route_meld_core_distance_with_fixed(
        *this, concealed, fixed_melds, mixed_shifted_chows_meld_cores());
  }
  if (fan == GbmjFan::MixedTripleChows) {
    return route_meld_core_distance_with_fixed(
        *this, concealed, fixed_melds, mixed_triple_chows_meld_cores());
  }
  if (fan == GbmjFan::MixedStraight) {
    return route_meld_core_distance_with_fixed(
        *this, concealed, fixed_melds, mixed_straight_meld_cores());
  }
  if (fan == GbmjFan::PureStraight) {
    return route_meld_core_distance_with_fixed(
        *this, concealed, fixed_melds, pure_straight_meld_cores());
  }
  if (fan == GbmjFan::PureShiftedChows) {
    return route_meld_core_distance_with_fixed(
        *this, concealed, fixed_melds, pure_shifted_chows_meld_cores());
  }
  if (fan == GbmjFan::KnittedStraight) {
    return knitted_straight_distance_with_fixed(*this, concealed, fixed_melds);
  }
  if (fan == GbmjFan::FullFlush) {
    return full_flush_distance_with_fixed(*this, concealed, fixed_melds);
  }
  if (fan == GbmjFan::HalfFlush) {
    return half_flush_distance_with_fixed(*this, concealed, owned, fixed_melds);
  }
  if (fan == GbmjFan::GreaterThanFive) {
    auto keep = [](const int tid) { return tid < 27 && rank_of(tid) >= 5; };
    if (!fixed_tiles_satisfy(fixed_melds, keep)) return 15;
    return regular_masked_distance_with_fixed(*this, concealed, fixed_count, keep);
  }
  if (fan == GbmjFan::LessThanFive) {
    auto keep = [](const int tid) { return tid < 27 && rank_of(tid) <= 3; };
    if (!fixed_tiles_satisfy(fixed_melds, keep)) return 15;
    return regular_masked_distance_with_fixed(*this, concealed, fixed_count, keep);
  }
  if (fan == GbmjFan::AllTypes) {
    return std::max(regular_distance_with_fixed(*this, concealed, fixed_count),
                    missing_all_types(owned));
  }
  if (fan == GbmjFan::NoPointHand || fan == GbmjFan::FullyMeldedHand) {
    return regular_distance_with_fixed(*this, concealed, fixed_count);
  }

  return 15;
}

Calsht::GbmjFanArray Calsht::calc_gbmj_fans(const std::array<int, NUM_TIDS>& t,
                                            const bool three_player) const
{
  GbmjFanArray ret{};
  for (int i = 0; i < NUM_GBMJ_MAIN_FANS; ++i) {
    ret[i] = calc_gbmj_fan(t, static_cast<GbmjFan>(i), three_player);
  }
  return ret;
}

Calsht::GbmjFanArray Calsht::calc_gbmj_fans(const std::array<int, NUM_TIDS>& concealed,
                                            const std::vector<GbmjMeld>& fixed_melds,
                                            const bool three_player) const
{
  GbmjFanArray ret{};
  for (int i = 0; i < NUM_GBMJ_MAIN_FANS; ++i) {
    ret[i] = calc_gbmj_fan(concealed, fixed_melds, static_cast<GbmjFan>(i), three_player);
  }
  return ret;
}

const std::array<const char*, Calsht::NUM_GBMJ_MAIN_FANS>& Calsht::gbmj_fan_names()
{
  static const std::array<const char*, NUM_GBMJ_MAIN_FANS> names{
    "mixed_shifted_chows",
    "all_types",
    "mixed_triple_chows",
    "mixed_straight",
    "half_flush",
    "pure_straight",
    "all_pungs",
    "pure_shifted_chows",
    "seven_pairs",
    "outside_hand",
    "full_flush",
    "no_point_hand",
    "greater_than_five",
    "knitted_straight",
    "less_than_five",
    "all_unrelated",
    "fully_melded_hand",
  };
  return names;
}

std::tuple<int, int> Calsht::operator()(const std::array<int, NUM_TIDS>& t,
                                        const int m,
                                        const int mode,
                                        const bool check_hand,
                                        const bool three_player) const
{
  if (check_hand) {
    int n = 0;

    for (int i = 0; i < NUM_TIDS; ++i) {
      if (t[i] < 0 || t[i] > 4) {
        std::ostringstream oss;
        oss << "Invalid number of hand's tiles at " << i << ": " << t[i];
        throw std::invalid_argument(oss.str());
      }

      ++n;
    }

    if (ENABLE_NYANTEN && n > 14) {
      throw std::invalid_argument("Invalid sum of hand's tiles: " + std::to_string(n));
    }

    if (m < 0 || m > 4) {
      throw std::invalid_argument("Invalid sum of hands's melds: " + std::to_string(m));
    }

    if (mode < 0 || mode > 7) {
      throw std::invalid_argument("Invalid caluculation mode: " + std::to_string(mode));
    }
  }

  std::tuple<int, int> ret{1024, 0};

  if (mode & 1) {
    if (int sht = calc_lh(t, m, three_player); sht < std::get<0>(ret)) {
      ret = {sht, 1};
    }
    else if (sht == std::get<0>(ret)) {
      std::get<1>(ret) |= 1;
    }
  }

  if ((mode & 2) && m == 4) {
    if (int sht = calc_sp(t, three_player); sht < std::get<0>(ret)) {
      ret = {sht, 2};
    }
    else if (sht == std::get<0>(ret)) {
      std::get<1>(ret) |= 2;
    }
  }

  if ((mode & 4) && m == 4) {
    if (int sht = calc_to(t); sht < std::get<0>(ret)) {
      ret = {sht, 4};
    }
    else if (sht == std::get<0>(ret)) {
      std::get<1>(ret) |= 4;
    }
  }

  return ret;
}
