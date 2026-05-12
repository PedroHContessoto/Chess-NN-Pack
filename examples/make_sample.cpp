// SPDX-License-Identifier: MIT
//
// examples/make_sample.cpp — produce a tiny CNNP V2 file for manual
// CLI testing.
//
// Usage:
//   make_sample <output_path>
//
// Writes 5 hand-crafted positions to <output_path> using the default
// Writer configuration (HalfP, max_feature_id=768, 1024-position blocks,
// minimal JSON metadata trailer).

#include "cnnp/cnnp.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: make_sample <output_path>\n";
        return 2;
    }

    // 5 positions covering a variety of input shapes:
    //   - large mid-game position (16 features)
    //   - opening-like (16 features, sequential ids)
    //   - endgame (8 features)
    //   - mid-endgame (10 features)
    //   - king-and-pawn (6 features)
    const std::vector<std::vector<std::uint16_t>> features = {
        {  1, 200, 320, 450,   0,  17, 134, 256,
         384, 512, 600, 700, 750, 760, 765, 767},
        {  5,  10,  20,  30,  40,  50,  60,  70,
          80,  90, 100, 110, 120, 130, 140, 150},
        {100, 200, 300, 400, 500, 600, 700, 760},
        { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500},
        {  1,   2,   3,   4,   5,   6},
    };
    const std::vector<std::int32_t> cps  = { 120,  -45,  500, -1500, 28000};
    const std::vector<std::int32_t> wdls = {   1,    0,   -1,     0,     1};
    const std::vector<std::uint8_t> stms = {   0,    1,    0,     1,     0};

    cnnp::Writer w;
    for (std::size_t i = 0; i < features.size(); ++i) {
        w.add(cps[i], wdls[i], stms[i],
              static_cast<std::uint8_t>(features[i].size()),
              std::span<const std::uint16_t>(features[i]));
    }
    w.finalize(argv[1]);

    std::cout << "Wrote " << features.size() << " positions to " << argv[1] << "\n";
    return 0;
}
