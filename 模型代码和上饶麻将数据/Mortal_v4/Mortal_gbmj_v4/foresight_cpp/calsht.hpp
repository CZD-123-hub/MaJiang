#ifndef CALSHT_HPP
#define CALSHT_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <tuple>
#include <vector>
#ifndef ENABLE_NYANTEN
#define ENABLE_NYANTEN false
#endif

class Calsht {
public:
  // [GBMJ main-fan shanten] High-frequency Chinese Standard Mahjong route IDs.
  // The returned value follows this library's distance convention:
  // complete hand is 0, one tile away is 1.
  static constexpr int NUM_GBMJ_MAIN_FANS = 17;
  enum class GbmjFan : int {
    MixedShiftedChows = 0,
    AllTypes,
    MixedTripleChows,
    MixedStraight,
    HalfFlush,
    PureStraight,
    AllPungs,
    PureShiftedChows,
    SevenPairs,
    OutsideHand,
    FullFlush,
    NoPointHand,
    GreaterThanFive,
    KnittedStraight,
    LessThanFive,
    AllUnrelated,
    FullyMeldedHand,
  };
  using GbmjFanArray = std::array<int, NUM_GBMJ_MAIN_FANS>;
  struct GbmjMeld {
    // [GBMJ fixed-meld route] Open/locked meld tiles. tile_count is 3 for
    // chow/pung and 4 for kong. Fixed melds cannot be regrouped by shanten.
    std::array<int, 4> tiles{};
    int tile_count = 3;
  };

private:
  using LVec = std::vector<uint8_t>;
  using RVec = std::vector<uint8_t>;
  using Iter = std::vector<RVec>::iterator;

  std::vector<RVec> mp1;
  std::vector<RVec> mp2;
  bool standard_tables_ready_ = false;

  RVec index1(int n) const;
  void add1(LVec& lhs, const RVec& rhs, int m) const;
  void add2(LVec& lhs, const RVec& rhs, int m) const;
  void read_file(Iter first, Iter last, std::filesystem::path file) const;
  int calc_one_meld_pair_distance(const std::array<int, 34>& t) const;
  int calc_required_core_distance(const std::array<int, 34>& t,
                                  const std::vector<int>& required) const;
  int calc_template_core_distance(const std::array<int, 34>& t,
                                  const std::vector<std::vector<int>>& cores) const;

public:
  Calsht()
      : mp1(ENABLE_NYANTEN ? 405350 : 1953125, RVec(10)),
        mp2(ENABLE_NYANTEN ? 43130 : 78125, RVec(10)) {}
  void initialize(const std::string& dir);
  int calc_lh(const std::array<int, 34>& t, int m, bool three_player = false) const;
  int calc_sp(const std::array<int, 34>& t, bool three_player = false) const;
  int calc_to(const std::array<int, 34>& t) const;
  int calc_gbmj_fan(const std::array<int, 34>& t,
                    GbmjFan fan,
                    bool three_player = false) const;
  int calc_gbmj_fan(const std::array<int, 34>& concealed,
                    const std::vector<GbmjMeld>& fixed_melds,
                    GbmjFan fan,
                    bool three_player = false) const;
  GbmjFanArray calc_gbmj_fans(const std::array<int, 34>& t,
                              bool three_player = false) const;
  GbmjFanArray calc_gbmj_fans(const std::array<int, 34>& concealed,
                              const std::vector<GbmjMeld>& fixed_melds,
                              bool three_player = false) const;
  // [GBMJ main-fan route table] Load/dump GBMJ route metadata.  The actual
  // heavy lookup payload is shanten_shu.bin/shanten_zi.bin; complete-hand
  // route-list scans are intentionally not used by calc_gbmj_fan().
  void initialize_gbmj_tables(const std::string& dir);
  static void make_gbmj_tables(const std::string& dir);
  static const std::array<const char*, NUM_GBMJ_MAIN_FANS>& gbmj_fan_names();
  std::tuple<int, int> operator()(const std::array<int, 34>& t,
                                  int m,
                                  int mode,
                                  bool check_hand = false,
                                  bool three_player = false) const;
};

#endif
