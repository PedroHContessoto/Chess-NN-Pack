// SPDX-License-Identifier: GPL-3.0-or-later
//
// tools/binpack_to_cnnp/binpack_to_cnnp.cpp
//
// Converts a Stockfish .binpack training file into a CNNP V2 file.
//
//   binpack_to_cnnp <in.binpack> <out.cnnp> [options]
//
// Options:
//   --max-positions N      stop after N kept positions (0 = unlimited, default)
//   --no-filter            skip the Stockfish-style filter (keep captures + in-check)
//   --metadata-json '..'   custom JSON trailer (default: minimal spec-compliant)
//   --report-every N       progress line every N read positions (default 1_000_000)
//   --max-mem-gb N         in-memory writer cap (default 32 GB; raise for ≥1B
//                          positions on high-RAM machines, e.g. --max-mem-gb 64)
//   --quiet                suppress progress reporting
//
// PERFORMANCE NOTE
// ────────────────
// This converter is single-threaded and runs at ~1.4M positions/s on a
// modern x86_64 (≈13 s for a 60 MB Linrock T77 binpack). Multi-threading
// is intentionally deferred — the binpack reader is single-threaded by
// design (delta-encoded stream), so the realistic ceiling for a producer/
// consumer pipeline is only ~2x. We may revisit later with a different
// approach (e.g., SIMD HalfP, faster vendored reader, chunk-level parallelism).
//
// LICENSE
// ───────
// This binary VENDORS the nnue-pytorch headers
// (`nnue_training_data_formats.h`, GPL-3.0-or-later). As a result, this
// executable is GPL-3.0-or-later. The cnnp library it links against
// remains MIT (one-way absorption is allowed).
//
// SEMANTICS NOTES
// ───────────────
// - Eval (`entry.score`): stored AS-IS from binpack. Binpacks use the
//   side-to-move POV. CNNP stores whatever it receives; the consumer
//   (training script) decides how to interpret it.
// - WDL (`entry.result`): white-POV in binpack and white-POV in CNNP
//   (spec §5.3). No flipping needed.
// - Features: full HalfP encoding (color*384 + piece_type*64 + square),
//   max_feature_id = 768. Kings are INCLUDED (HalfKA-flavored variants
//   would exclude them; pure HalfP per spec §5.4 includes them).
// - Filter: matches the standard Stockfish-style skip (captures +
//   in-check positions). Roughly ~25-30% of positions get filtered.

#include "cnnp/cnnp.hpp"

// Vendored from nnue-pytorch (GPL-3.0). Contains BOTH the binpack
// reader and the chess primitives (Position/Piece/Square/Color/...).
#include "nnue_training_data_formats.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int EXIT_OK         = 0;
constexpr int EXIT_USAGE      = 1;
constexpr int EXIT_FILE       = 2;
constexpr int EXIT_CONVERSION = 3;

void print_usage(std::ostream& os) {
    os <<
        "binpack_to_cnnp — convert Stockfish .binpack into CNNP V2.\n"
        "\n"
        "Usage:\n"
        "  binpack_to_cnnp <in.binpack> <out.cnnp> [options]\n"
        "\n"
        "Options:\n"
        "  --max-positions N      stop after N KEPT positions (0 = no cap)\n"
        "  --no-filter            keep captures and in-check positions\n"
        "  --metadata-json '..'   custom UTF-8 JSON trailer\n"
        "  --report-every N       progress line every N read positions (0 = off)\n"
        "  --max-mem-gb N         writer in-memory cap in gigabytes (default 32;\n"
        "                         raise for ≥1B positions on high-RAM machines)\n"
        "  --quiet                suppress all progress output\n"
        "  -h, --help             show this help\n"
        "\n"
        "Filter (default ON, matches Stockfish-style skip predicate):\n"
        "  - skip positions whose best move is a capture\n"
        "  - skip positions where the side-to-move is in check\n"
        "\n"
        "Exit codes:\n"
        "  0  success    1  usage error\n"
        "  2  file/IO    3  conversion error (validator failed, etc.)\n";
}

bool parse_u64(std::string_view s, std::uint64_t& out) {
    try { out = std::stoull(std::string(s)); return true; }
    catch (...) { return false; }
}

}  // anonymous namespace

int main(int argc, char** argv) {
    // ─── Parse args ───────────────────────────────────────────────────────────
    if (argc < 2) {
        print_usage(std::cerr);
        return EXIT_USAGE;
    }
    {
        const std::string_view a1(argv[1]);
        if (a1 == "-h" || a1 == "--help" || a1 == "help") {
            print_usage(std::cout);
            return EXIT_OK;
        }
    }
    if (argc < 3) {
        std::cerr << "Error: expected <in.binpack> <out.cnnp>\n\n";
        print_usage(std::cerr);
        return EXIT_USAGE;
    }

    std::string   in_path  = argv[1];
    std::string   out_path = argv[2];
    std::uint64_t max_positions = 0;
    bool          filter = true;
    std::uint64_t report_every = 1'000'000;
    bool          quiet = false;
    std::string   metadata_json;  // empty → Writer auto-fills with default
    std::uint64_t max_mem_gb = 32;  // default cap for the in-memory writer

    for (int i = 3; i < argc; ++i) {
        const std::string_view a(argv[i]);
        auto need_arg = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " requires a value\n";
                std::exit(EXIT_USAGE);
            }
            return argv[++i];
        };
        if (a == "--max-positions") {
            if (!parse_u64(need_arg("--max-positions"), max_positions)) {
                std::cerr << "Error: --max-positions expects a non-negative integer\n";
                return EXIT_USAGE;
            }
        } else if (a == "--no-filter") {
            filter = false;
        } else if (a == "--metadata-json") {
            metadata_json = need_arg("--metadata-json");
        } else if (a == "--report-every") {
            if (!parse_u64(need_arg("--report-every"), report_every)) {
                std::cerr << "Error: --report-every expects a non-negative integer\n";
                return EXIT_USAGE;
            }
        } else if (a == "--max-mem-gb") {
            if (!parse_u64(need_arg("--max-mem-gb"), max_mem_gb) || max_mem_gb == 0) {
                std::cerr << "Error: --max-mem-gb expects a positive integer\n";
                return EXIT_USAGE;
            }
        } else if (a == "--quiet") {
            quiet = true;
        } else {
            std::cerr << "Error: unknown option '" << a << "'\n\n";
            print_usage(std::cerr);
            return EXIT_USAGE;
        }
    }

    // ─── Open the binpack ────────────────────────────────────────────────────
    std::unique_ptr<binpack::CompressedTrainingDataEntryReader> reader;
    try {
        reader = std::make_unique<binpack::CompressedTrainingDataEntryReader>(in_path);
    } catch (const std::exception& e) {
        std::cerr << "Error opening binpack '" << in_path << "': " << e.what() << "\n";
        return EXIT_FILE;
    }

    // ─── Set up the CNNP writer ──────────────────────────────────────────────
    cnnp::WriterConfig cfg;
    if (!metadata_json.empty()) cfg.metadata_json = metadata_json;
    // Defensive cap (configurable). Catches accidental mass-conversion
    // before bad_alloc. Raise via --max-mem-gb on high-RAM machines.
    cfg.max_in_memory_bytes = max_mem_gb * (1ull << 30);
    if (!quiet) {
        std::cerr << "Writer in-memory cap: " << max_mem_gb
                  << " GB (override with --max-mem-gb)\n";
    }

    cnnp::Writer writer;
    try {
        writer = cnnp::Writer(cfg);
    } catch (const cnnp::WriteError& e) {
        std::cerr << "Error initialising writer: " << e.what() << "\n";
        return EXIT_CONVERSION;
    }

    // ─── Convert loop ─────────────────────────────────────────────────────────
    std::vector<std::uint16_t> features;
    features.reserve(32);

    std::uint64_t total_read              = 0;
    std::uint64_t total_kept              = 0;
    std::uint64_t skipped_captures        = 0;
    std::uint64_t skipped_in_check        = 0;
    std::uint64_t skipped_writer_rejected = 0;

    const auto t_start = std::chrono::steady_clock::now();
    auto last_report   = t_start;

    while (reader->hasNext()) {
        binpack::TrainingDataEntry entry;
        try {
            entry = reader->next();
        } catch (const std::exception& e) {
            std::cerr << "\nError reading position " << total_read
                      << ": " << e.what() << "\n";
            return EXIT_FILE;
        }
        ++total_read;

        if (filter) {
            if (entry.isCapturingMove()) {
                ++skipped_captures;
                goto progress;
            }
            if (entry.isInCheck()) {
                ++skipped_in_check;
                goto progress;
            }
        }

        // ── HalfP feature extraction ──────────────────────────────────────────
        features.clear();
        for (int sqi = 0; sqi < 64; ++sqi) {
            const chess::Square sq{sqi};
            const chess::Piece  pc = entry.pos.pieceAt(sq);
            if (pc == chess::Piece::none()) continue;

            const std::uint16_t color =
                static_cast<std::uint16_t>(static_cast<int>(pc.color()));
            const std::uint16_t pt =
                static_cast<std::uint16_t>(static_cast<int>(pc.type()));
            const std::uint16_t sq_u =
                static_cast<std::uint16_t>(sqi);

            features.push_back(static_cast<std::uint16_t>(
                color * 384u + pt * 64u + sq_u));
        }

        if (features.size() < 2 || features.size() > 32) {
            // Illegal chess (no kings, or 33+ pieces) — skip defensively.
            ++skipped_writer_rejected;
            goto progress;
        }

        try {
            const std::uint8_t stm =
                static_cast<std::uint8_t>(static_cast<int>(entry.pos.sideToMove()));
            writer.add(
                static_cast<std::int32_t>(entry.score),
                static_cast<std::int32_t>(entry.result),
                stm,
                static_cast<std::uint8_t>(features.size()),
                std::span<const std::uint16_t>(features));
            ++total_kept;
        } catch (const cnnp::WriteError& e) {
            ++skipped_writer_rejected;
            if (!quiet && skipped_writer_rejected <= 5) {
                std::cerr << "  skip pos " << total_read
                          << ": " << e.what() << "\n";
            }
            goto progress;
        }

        if (max_positions > 0 && total_kept >= max_positions) break;

    progress:
        if (!quiet && report_every > 0 && (total_read % report_every == 0)) {
            const auto now = std::chrono::steady_clock::now();
            const auto dt  = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - last_report).count();
            const double rate = (dt > 0)
                ? (1000.0 * report_every / static_cast<double>(dt))
                : 0.0;
            std::cerr << "[read=" << total_read
                      << " kept=" << total_kept
                      << " rate=" << static_cast<std::uint64_t>(rate)
                      << " pos/s]\n";
            last_report = now;
        }
    }

    if (total_kept == 0) {
        std::cerr << "Error: no positions kept (read " << total_read
                  << ", all filtered or rejected)\n";
        return EXIT_CONVERSION;
    }

    // ─── Finalise ─────────────────────────────────────────────────────────────
    if (!quiet) {
        std::cerr << "\nFinalising " << total_kept
                  << " positions → " << out_path << " ...\n";
    }
    try {
        writer.finalize(out_path);
    } catch (const cnnp::WriteError& e) {
        std::cerr << "Error finalizing CNNP file: " << e.what() << "\n";
        return EXIT_CONVERSION;
    } catch (const cnnp::ValidationError& e) {
        std::cerr << "Error: writer produced an invalid file: " << e.what() << "\n";
        return EXIT_CONVERSION;
    }

    // ─── Summary ──────────────────────────────────────────────────────────────
    const auto t_end = std::chrono::steady_clock::now();
    const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             t_end - t_start).count();
    const double pct_kept = total_read > 0
        ? (100.0 * static_cast<double>(total_kept) / static_cast<double>(total_read))
        : 0.0;

    std::cout << "\n--- Summary ---\n"
              << "  Source        : " << in_path << "\n"
              << "  Destination   : " << out_path << "\n"
              << "  Read          : " << total_read << "\n"
              << "  Kept          : " << total_kept
              << "  (" << pct_kept << "%)\n"
              << "  Skipped:\n"
              << "    captures    : " << skipped_captures << "\n"
              << "    in-check    : " << skipped_in_check << "\n"
              << "    rejected    : " << skipped_writer_rejected << "\n"
              << "  Wall time     : " << (wall_ms / 1000.0) << " s\n";
    if (wall_ms > 0) {
        std::cout << "  Throughput    : "
                  << static_cast<std::uint64_t>(
                         1000.0 * total_read / static_cast<double>(wall_ms))
                  << " pos/s (read)\n";
    }
    return EXIT_OK;
}
