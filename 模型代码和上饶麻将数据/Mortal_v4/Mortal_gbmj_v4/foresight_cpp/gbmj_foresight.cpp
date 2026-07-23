#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "calsht.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kTileCount = 34;
constexpr int kRouteCount = Calsht::NUM_GBMJ_MAIN_FANS;
constexpr int kGroupCount = 7;
constexpr int kTopDiscardCount = 4;

using Hand = std::array<int, kTileCount>;
using Distances = Calsht::GbmjFanArray;
using FixedMelds = std::vector<Calsht::GbmjMeld>;

constexpr std::array<int, kRouteCount> kFanValues = {
    6,   // MixedShiftedChows
    6,   // AllTypes
    8,   // MixedTripleChows
    8,   // MixedStraight
    6,   // HalfFlush
    16,  // PureStraight
    6,   // AllPungs
    16,  // PureShiftedChows
    24,  // SevenPairs
    4,   // OutsideHand
    24,  // FullFlush
    8,   // NoPointHand
    12,  // GreaterThanFive
    12,  // KnittedStraight
    12,  // LessThanFive
    12,  // AllUnrelated
    6,   // FullyMeldedHand
};

constexpr std::array<std::array<int, 6>, kGroupCount> kRouteGroups = {{
    {{8, -1, -1, -1, -1, -1}},       // seven_pairs
    {{13, 15, -1, -1, -1, -1}},      // knitted_unrelated
    {{6, -1, -1, -1, -1, -1}},       // pung_route
    {{4, 10, -1, -1, -1, -1}},       // flush_route
    {{5, 7, -1, -1, -1, -1}},        // pure_sequence
    {{0, 2, 3, -1, -1, -1}},         // mixed_sequence
    {{1, 9, 11, 12, 14, 16}},        // terminal_value
}};

struct EvalResult {
    double score = 0.0;
    Distances distances{};
    double best_distance = 14.0;
};

std::string make_key(const Hand& hand, const FixedMelds& fixed_melds) {
    std::string key;
    key.reserve(kTileCount + fixed_melds.size() * 5 + 1);
    for (int value : hand) {
        key.push_back(static_cast<char>(std::max(0, std::min(4, value))));
    }
    key.push_back(static_cast<char>(fixed_melds.size()));
    for (const auto& meld : fixed_melds) {
        key.push_back(static_cast<char>(meld.tile_count));
        for (int i = 0; i < meld.tile_count && i < 4; ++i) {
            key.push_back(static_cast<char>(meld.tiles[i]));
        }
    }
    return key;
}

class ForesightEngine {
public:
    ForesightEngine(std::string table_dir, int cache_limit)
        : table_dir_(std::move(table_dir)),
          cache_limit_(std::max(0, cache_limit)) {
        calsht_.initialize_gbmj_tables(table_dir_);
    }

    EvalResult evaluate(const Hand& hand, const FixedMelds& fixed_melds) {
        const std::string key = make_key(hand, fixed_melds);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        EvalResult result;
        result.distances = calsht_.calc_gbmj_fans(hand, fixed_melds);

        double score = 0.0;
        int best_distance = 15;
        for (int i = 0; i < kRouteCount; ++i) {
            const int distance = std::max(0, result.distances[i]);
            score += std::sqrt(static_cast<double>(kFanValues[i]) / (static_cast<double>(distance) + 1.0));
            best_distance = std::min(best_distance, distance);
        }

        result.score = score;
        result.best_distance = static_cast<double>(best_distance);

        if (cache_limit_ > 0) {
            if (static_cast<int>(cache_.size()) >= cache_limit_) {
                // [V4 foresight C++ batch] Keep worker memory bounded.  A clear
                // is cheaper and safer than an exact LRU for DataLoader workers.
                cache_.clear();
            }
            cache_.emplace(key, result);
        }
        return result;
    }

private:
    std::string table_dir_;
    int cache_limit_ = 0;
    Calsht calsht_;
    std::unordered_map<std::string, EvalResult> cache_;
};

std::map<std::string, std::unique_ptr<ForesightEngine>>& engines() {
    static std::map<std::string, std::unique_ptr<ForesightEngine>> instance;
    return instance;
}

ForesightEngine& get_engine(const std::string& table_dir, int cache_limit) {
    const std::string key = table_dir + "|" + std::to_string(cache_limit);
    auto& map_ref = engines();
    auto it = map_ref.find(key);
    if (it != map_ref.end()) {
        return *(it->second);
    }
    std::unique_ptr<ForesightEngine> engine(new ForesightEngine(table_dir, cache_limit));
    ForesightEngine& ref = *engine;
    map_ref.emplace(key, std::move(engine));
    return ref;
}

bool parse_hand(PyObject* obj, Hand& hand) {
    PyObject* seq = PySequence_Fast(obj, "hand must be a sequence of 34 integers");
    if (seq == nullptr) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size < kTileCount) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "hand must contain at least 34 values");
        return false;
    }
    for (int i = 0; i < kTileCount; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject* number = PyNumber_Long(item);
        if (number == nullptr) {
            Py_DECREF(seq);
            return false;
        }
        const long raw = PyLong_AsLong(number);
        Py_DECREF(number);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        hand[i] = static_cast<int>(std::max<long>(0, std::min<long>(4, raw)));
    }
    Py_DECREF(seq);
    return true;
}

bool parse_flags(PyObject* obj, std::array<bool, kTileCount>& flags) {
    PyObject* seq = PySequence_Fast(obj, "discard_candidates must be a sequence of 34 bool-like values");
    if (seq == nullptr) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size < kTileCount) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "discard_candidates must contain at least 34 values");
        return false;
    }
    for (int i = 0; i < kTileCount; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const int truth = PyObject_IsTrue(item);
        if (truth < 0) {
            Py_DECREF(seq);
            return false;
        }
        flags[i] = truth != 0;
    }
    Py_DECREF(seq);
    return true;
}

bool parse_fixed_melds(PyObject* obj, FixedMelds& fixed_melds) {
    if (obj == nullptr || obj == Py_None) {
        return true;
    }

    PyObject* outer = PySequence_Fast(obj, "fixed_melds must be a sequence of meld tile-id sequences");
    if (outer == nullptr) {
        return false;
    }

    const Py_ssize_t meld_count = PySequence_Fast_GET_SIZE(outer);
    fixed_melds.reserve(static_cast<size_t>(meld_count));
    for (Py_ssize_t m = 0; m < meld_count; ++m) {
        PyObject* meld_obj = PySequence_Fast_GET_ITEM(outer, m);
        PyObject* meld_seq = PySequence_Fast(meld_obj, "each fixed meld must be a sequence");
        if (meld_seq == nullptr) {
            Py_DECREF(outer);
            return false;
        }
        const Py_ssize_t tile_count = PySequence_Fast_GET_SIZE(meld_seq);
        if (tile_count != 3 && tile_count != 4) {
            Py_DECREF(meld_seq);
            Py_DECREF(outer);
            PyErr_SetString(PyExc_ValueError, "each fixed meld must contain 3 or 4 tile ids");
            return false;
        }

        Calsht::GbmjMeld meld;
        meld.tile_count = static_cast<int>(tile_count);
        for (Py_ssize_t i = 0; i < tile_count; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(meld_seq, i);
            PyObject* number = PyNumber_Long(item);
            if (number == nullptr) {
                Py_DECREF(meld_seq);
                Py_DECREF(outer);
                return false;
            }
            const long raw = PyLong_AsLong(number);
            Py_DECREF(number);
            if (PyErr_Occurred() || raw < 0 || raw >= kTileCount) {
                Py_DECREF(meld_seq);
                Py_DECREF(outer);
                PyErr_SetString(PyExc_ValueError, "fixed meld tile id must be in [0, 33]");
                return false;
            }
            meld.tiles[static_cast<size_t>(i)] = static_cast<int>(raw);
        }
        fixed_melds.push_back(meld);
        Py_DECREF(meld_seq);
    }

    Py_DECREF(outer);
    return true;
}

std::pair<std::vector<double>, std::vector<int>> compute_features(
    ForesightEngine& engine,
    const Hand& hand,
    const FixedMelds& fixed_melds,
    const std::array<bool, kTileCount>& discard_candidates) {
    const EvalResult current = engine.evaluate(hand, fixed_melds);

    std::vector<double> route_values;
    route_values.reserve(kGroupCount);
    for (int group = 0; group < kGroupCount; ++group) {
        int best = 15;
        for (int route : kRouteGroups[group]) {
            if (route < 0) {
                break;
            }
            best = std::min(best, std::max(0, current.distances[static_cast<size_t>(route)]));
        }
        route_values.push_back(1.0 / (static_cast<double>(best) + 1.0));
    }

    struct Candidate {
        double score;
        double best_distance;
        int tile;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(kTileCount);
    for (int tid = 0; tid < kTileCount; ++tid) {
        if (!discard_candidates[tid] || hand[tid] <= 0) {
            continue;
        }
        Hand next = hand;
        next[tid] -= 1;
        const EvalResult next_eval = engine.evaluate(next, fixed_melds);
        candidates.push_back({next_eval.score, next_eval.best_distance, tid});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        if (lhs.best_distance != rhs.best_distance) {
            return lhs.best_distance < rhs.best_distance;
        }
        return lhs.tile < rhs.tile;
    });

    std::vector<int> top_discards;
    const int limit = std::min<int>(kTopDiscardCount, static_cast<int>(candidates.size()));
    top_discards.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        top_discards.push_back(candidates[static_cast<size_t>(i)].tile);
    }
    return {route_values, top_discards};
}

PyObject* py_compute_foresight(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* hand_obj = nullptr;
    PyObject* fixed_melds_obj = nullptr;
    PyObject* discard_obj = nullptr;
    const char* table_dir = "data";
    int cache_limit = 200000;

    static const char* kwlist[] = {
        "hand",
        "fixed_melds",
        "discard_candidates",
        "table_dir",
        "cache_limit",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOO|si",
            const_cast<char**>(kwlist),
            &hand_obj,
            &fixed_melds_obj,
            &discard_obj,
            &table_dir,
            &cache_limit)) {
        return nullptr;
    }

    Hand hand{};
    FixedMelds fixed_melds;
    std::array<bool, kTileCount> discard_candidates{};
    if (!parse_hand(hand_obj, hand) ||
        !parse_fixed_melds(fixed_melds_obj, fixed_melds) ||
        !parse_flags(discard_obj, discard_candidates)) {
        return nullptr;
    }

    try {
        ForesightEngine& engine = get_engine(std::string(table_dir), cache_limit);
        const auto result = compute_features(engine, hand, fixed_melds, discard_candidates);

        PyObject* route_list = PyList_New(static_cast<Py_ssize_t>(result.first.size()));
        if (route_list == nullptr) {
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(result.first.size()); ++i) {
            PyObject* value = PyFloat_FromDouble(result.first[static_cast<size_t>(i)]);
            if (value == nullptr) {
                Py_DECREF(route_list);
                return nullptr;
            }
            PyList_SET_ITEM(route_list, i, value);
        }

        PyObject* discard_list = PyList_New(static_cast<Py_ssize_t>(result.second.size()));
        if (discard_list == nullptr) {
            Py_DECREF(route_list);
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(result.second.size()); ++i) {
            PyObject* value = PyLong_FromLong(result.second[static_cast<size_t>(i)]);
            if (value == nullptr) {
                Py_DECREF(route_list);
                Py_DECREF(discard_list);
                return nullptr;
            }
            PyList_SET_ITEM(discard_list, i, value);
        }

        PyObject* tuple = PyTuple_New(2);
        if (tuple == nullptr) {
            Py_DECREF(route_list);
            Py_DECREF(discard_list);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, 0, route_list);
        PyTuple_SET_ITEM(tuple, 1, discard_list);
        return tuple;
    }
    catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

PyObject* py_clear_cache(PyObject*, PyObject*) {
    engines().clear();
    Py_RETURN_NONE;
}

PyMethodDef kMethods[] = {
    {"compute_foresight", reinterpret_cast<PyCFunction>(py_compute_foresight), METH_VARARGS | METH_KEYWORDS,
     "compute_foresight(hand, fixed_melds, discard_candidates, table_dir='data', cache_limit=200000)"},
    {"clear_cache", py_clear_cache, METH_NOARGS, "Clear cached C++ foresight engines."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "gbmj_foresight_cpp",
    "C++ batch implementation of GBMJ v4 foresight feature calculation.",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_gbmj_foresight_cpp(void) {
    return PyModule_Create(&kModule);
}
