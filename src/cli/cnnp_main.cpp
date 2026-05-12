// SPDX-License-Identifier: MIT
//
// cnnp/cli/cnnp_main.cpp — command-line tool entry point.
//
// Usage:
//   cnnp <subcommand> <file> [options]
//
// Subcommands:
//   validate <file>           Run full spec validation. Exit 0 on pass,
//                             non-zero on failure (with reason on stderr).
//   inspect  <file>           Print header summary + WDL/feature stats.
//   dump     <file> [opts]    Print a range of positions in human-readable
//                             form. Default range: 0:5.
//                             --range START:END   pick a custom range.
//   help                      Print this usage block.
//
// Exit codes:
//   0  success
//   1  validation failure (header/parse/validator error)
//   2  usage error (bad arguments)
//   3  file/IO error (mmap failure, missing file, etc.)

#include "cnnp/cnnp.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

namespace {

constexpr int EXIT_OK         = 0;
constexpr int EXIT_VALIDATION = 1;
constexpr int EXIT_USAGE      = 2;
constexpr int EXIT_FILE       = 3;

void print_usage(std::ostream& os) {
    os <<
        "cnnp — Chess-NN-Pack CLI (V2)\n"
        "\n"
        "Usage:\n"
        "  cnnp <subcommand> <file> [options]\n"
        "\n"
        "Subcommands:\n"
        "  validate <file>           Run full spec validation.\n"
        "  inspect  <file>           Print header summary + stats.\n"
        "  dump     <file> [opts]    Print positions in human-readable form.\n"
        "      --range START:END     Position range (default: 0:5)\n"
        "  help                      Show this help.\n"
        "\n"
        "Exit codes:\n"
        "  0  success    1  validation failure\n"
        "  2  usage err  3  file/IO error\n";
}

// ─── validate ────────────────────────────────────────────────────────────────

// Full JSON validation of the metadata trailer (spec §6 + §12 require
// it to be valid UTF-8 JSON containing the `format` and `layout` fields).
// Uses nlohmann/json — a CLI-only dependency; the core library remains
// dependency-free per spec §12.
void validate_metadata_json(std::span<const std::byte> md) {
    if (md.empty()) {
        throw cnnp::ValidationError(
            "metadata trailer is empty (spec §6 requires UTF-8 JSON)");
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(md.begin(), md.end());
    } catch (const nlohmann::json::parse_error& e) {
        throw cnnp::ValidationError(
            std::string("metadata is not valid JSON: ") + e.what());
    }
    if (!j.is_object()) {
        throw cnnp::ValidationError(
            "metadata is JSON but not an object (spec requires {...})");
    }

    const auto fmt_it = j.find("format");
    if (fmt_it == j.end() || !fmt_it->is_string()) {
        throw cnnp::ValidationError(
            "metadata missing required string field \"format\"");
    }
    if (fmt_it->get<std::string>() != "cnnp_sparse_v2") {
        throw cnnp::ValidationError(
            "metadata \"format\" must equal \"cnnp_sparse_v2\"; got \""
            + fmt_it->get<std::string>() + "\"");
    }

    const auto layout_it = j.find("layout");
    if (layout_it == j.end() || !layout_it->is_string()) {
        throw cnnp::ValidationError(
            "metadata missing required string field \"layout\"");
    }
    if (layout_it->get<std::string>() != "single_file") {
        throw cnnp::ValidationError(
            "metadata \"layout\" must equal \"single_file\" in V2; got \""
            + layout_it->get<std::string>() + "\"");
    }
}

int cmd_validate(const std::filesystem::path& path) {
    // Reader::open runs parse_header + validate_full (binary layer).
    // Then we additionally parse and check the JSON metadata trailer.
    auto r = cnnp::Reader::open(path);
    validate_metadata_json(r.metadata());

    std::cout << "OK  " << path.string()
              << "  num_positions=" << r.num_positions()
              << "  num_blocks="    << r.num_blocks()
              << "  num_features="  << r.header().num_features_total
              << "  metadata="      << r.header().metadata_length << "B\n";
    return EXIT_OK;
}

// ─── inspect ─────────────────────────────────────────────────────────────────

const char* feature_set_name(cnnp::FeatureSet fs) {
    switch (fs) {
        case cnnp::FeatureSet::HalfP:       return "HalfP";
        case cnnp::FeatureSet::HalfKAv2_hm: return "HalfKAv2_hm";
        case cnnp::FeatureSet::HalfKA:      return "HalfKA";
    }
    return "?";
}

int cmd_inspect(const std::filesystem::path& path) {
    auto r = cnnp::Reader::open(path);
    const auto& h = r.header();

    std::cout << "─── CNNP V" << h.version << " — " << path.string() << " ───\n";

    std::cout << "Identity:\n"
              << "  magic                : CNN2\n"
              << "  version              : " << h.version << "\n"
              << "  header_size          : " << h.header_size << "\n"
              << "  endian               : little\n"
              << "  layout_kind          : single_file\n";

    std::cout << "Semantics:\n"
              << "  feature_set          : " << feature_set_name(h.feature_set) << "\n"
              << "  count_semantics      : piece_count == nnz\n"
              << "  eval_encoding        : int16 fixed-normalized\n"
              << "  flags_encoding       : stm | (count-2)<<1\n";

    std::cout << "Sizes:\n"
              << "  num_positions        : " << h.num_positions << "\n"
              << "  num_features_total   : " << h.num_features_total << "\n"
              << "  max_feature_id       : " << h.max_feature_id << "\n"
              << "  num_blocks           : " << h.num_blocks << "\n"
              << "  block_size           : " << h.block_size << "\n"
              << "  max_count            : " << static_cast<int>(h.max_count) << "\n";

    std::cout << "Eval params:\n"
              << "  fixed_scale          : " << h.fixed_scale << "\n"
              << "  normal_eval_clip     : " << h.normal_eval_clip << "\n"
              << "  storage_target_clip  : " << h.storage_target_clip << "\n";

    std::cout << "Layout (byte offsets):\n"
              << "  flags_offset         : " << h.flags_offset << "\n"
              << "  eval_offset          : " << h.eval_offset << "\n"
              << "  wdl_offset           : " << h.wdl_offset << "\n"
              << "  block_anchors_offset : " << h.block_anchors_offset << "\n"
              << "  block_prefix_offset  : " << h.block_prefix_offset << "\n"
              << "  w_flat_offset        : " << h.w_flat_offset << "\n"
              << "  metadata_offset      : " << h.metadata_offset << "\n"
              << "  metadata_length      : " << h.metadata_length << "\n";

    // ── WDL distribution
    auto wdls = r.wdl_array();
    std::uint64_t w = 0, d = 0, l = 0;
    for (auto v : wdls) {
        if (v == 1)       ++w;
        else if (v == 0)  ++d;
        else              ++l;  // -1
    }
    const double total = wdls.empty() ? 1.0 : static_cast<double>(wdls.size());
    std::cout << "WDL distribution:\n"
              << std::fixed << std::setprecision(2)
              << "  white wins           : " << w << "  (" << (100.0 * w / total) << "%)\n"
              << "  draws                : " << d << "  (" << (100.0 * d / total) << "%)\n"
              << "  black wins           : " << l << "  (" << (100.0 * l / total) << "%)\n";

    // ── Feature density
    std::cout << "Feature density:\n"
              << "  avg features/pos     : "
              << (static_cast<double>(h.num_features_total) /
                  std::max<double>(static_cast<double>(h.num_positions), 1.0))
              << "\n";
    std::cout.unsetf(std::ios_base::floatfield);

    // ── File size
    std::cout << "File:\n"
              << "  size_on_disk         : " << std::filesystem::file_size(path) << " bytes\n";

    // ── Metadata trailer (raw bytes; core does not parse JSON)
    if (h.metadata_length > 0) {
        std::cout << "Metadata (UTF-8 JSON, " << h.metadata_length << " bytes):\n  ";
        auto md = r.metadata();
        std::cout.write(reinterpret_cast<const char*>(md.data()),
                        static_cast<std::streamsize>(md.size()));
        std::cout << "\n";
    } else {
        std::cout << "Metadata             : (empty)\n";
    }

    return EXIT_OK;
}

// ─── dump ────────────────────────────────────────────────────────────────────

int cmd_dump(const std::filesystem::path& path,
             std::uint64_t start, std::uint64_t end) {
    auto r = cnnp::Reader::open(path);

    end = std::min(end, r.num_positions());
    if (start >= end) {
        std::cerr << "dump: empty range " << start << ":" << end
                  << " (num_positions=" << r.num_positions() << ")\n";
        return EXIT_USAGE;
    }

    for (std::uint64_t i = start; i < end; ++i) {
        auto v = r.at(i);
        std::cout << "[" << i << "] "
                  << "stm=" << static_cast<int>(v.stm)
                  << " pc="  << static_cast<int>(v.piece_count)
                  << " wdl=" << static_cast<int>(v.wdl)
                  << " eval=" << v.eval_normalized
                  << " features=[";
        for (std::size_t k = 0; k < v.features.size(); ++k) {
            if (k) std::cout << ",";
            std::cout << v.features[k];
        }
        std::cout << "]\n";
    }

    return EXIT_OK;
}

// ─── argv parsing helpers ────────────────────────────────────────────────────

bool parse_range(std::string_view s, std::uint64_t& start, std::uint64_t& end) {
    const auto colon = s.find(':');
    if (colon == std::string_view::npos) return false;
    try {
        start = std::stoull(std::string(s.substr(0, colon)));
        end   = std::stoull(std::string(s.substr(colon + 1)));
    } catch (...) {
        return false;
    }
    return start <= end;
}

}  // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(std::cerr);
        return EXIT_USAGE;
    }

    const std::string_view cmd(argv[1]);
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_usage(std::cout);
        return EXIT_OK;
    }

    if (argc < 3) {
        std::cerr << "Error: missing <file> for subcommand '" << cmd << "'\n\n";
        print_usage(std::cerr);
        return EXIT_USAGE;
    }

    const std::filesystem::path path(argv[2]);

    try {
        if (cmd == "validate") return cmd_validate(path);
        if (cmd == "inspect")  return cmd_inspect(path);
        if (cmd == "dump") {
            std::uint64_t start = 0, end = 5;
            for (int i = 3; i < argc; ++i) {
                const std::string_view a(argv[i]);
                if (a == "--range") {
                    if (i + 1 >= argc) {
                        std::cerr << "Error: --range requires START:END\n";
                        return EXIT_USAGE;
                    }
                    if (!parse_range(argv[++i], start, end)) {
                        std::cerr << "Error: invalid --range value '"
                                  << argv[i] << "'\n";
                        return EXIT_USAGE;
                    }
                } else {
                    std::cerr << "Error: unknown 'dump' option '" << a << "'\n";
                    return EXIT_USAGE;
                }
            }
            return cmd_dump(path, start, end);
        }

        std::cerr << "Error: unknown subcommand '" << cmd << "'\n\n";
        print_usage(std::cerr);
        return EXIT_USAGE;
    } catch (const cnnp::MmapError& e) {
        std::cerr << "Error (mmap): " << e.what() << "\n";
        return EXIT_FILE;
    } catch (const cnnp::ParseError& e) {
        std::cerr << "Error (header parse): " << e.what() << "\n";
        return EXIT_VALIDATION;
    } catch (const cnnp::ValidationError& e) {
        std::cerr << "Error (validator): " << e.what() << "\n";
        return EXIT_VALIDATION;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FILE;
    }
}
