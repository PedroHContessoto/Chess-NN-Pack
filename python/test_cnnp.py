# SPDX-License-Identifier: MIT
#
# python/test_cnnp.py — smoke tests for the cnnp Python module.
#
# Run from build/python/ (where cnnp.*.pyd lives):
#     cd build/python
#     python ../../python/test_cnnp.py
#
# Or with PYTHONPATH:
#     PYTHONPATH=build/python python python/test_cnnp.py

import json
import os
import sys
import tempfile
import traceback

import numpy as np

import cnnp

# ─── Helpers ─────────────────────────────────────────────────────────────────


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def record(self, name, ok, msg=""):
        if ok:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed += 1
            self.failures.append((name, msg))
            print(f"  FAIL  {name}: {msg}")


def section(title):
    print(f"\n-- {title} --")


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_module_constants(r):
    section("Module constants")
    r.record("VERSION == 2",            cnnp.VERSION == 2)
    r.record("HEADER_SIZE == 256",      cnnp.HEADER_SIZE == 256)
    r.record("BLOCK_SIZE == 1024",      cnnp.BLOCK_SIZE == 1024)
    r.record("MAX_COUNT == 32",         cnnp.MAX_COUNT == 32)
    r.record("COUNT_BASE == 2",         cnnp.COUNT_BASE == 2)
    r.record("FIXED_SCALE == 3000.0",   cnnp.FIXED_SCALE == 3000.0)
    r.record("MAGIC == b'CNN2'",        cnnp.MAGIC == b"CNN2")
    r.record("FeatureSet.HalfP exists", cnnp.FeatureSet.HalfP.value == 1)


def make_sample(path):
    w = cnnp.Writer()  # default config
    feats = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    for i in range(4):
        w.add(score_cp=i * 50,
              wdl=(i % 3) - 1,
              stm=i % 2,
              piece_count=8,
              features=feats)
    w.finalize(path)


def test_writer_reader_roundtrip(r):
    section("Writer → Reader round-trip")
    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name

    try:
        make_sample(path)

        rdr = cnnp.Reader(path)
        r.record("num_positions == 4",  rdr.num_positions == 4)
        r.record("num_blocks == 1",     rdr.num_blocks == 1)
        r.record("len(reader) == 4",    len(rdr) == 4)

        for i in range(4):
            pos = rdr.at(i)
            r.record(f"at({i}).stm",         pos.stm == (i % 2))
            r.record(f"at({i}).piece_count", pos.piece_count == 8)
            r.record(f"at({i}).wdl",         pos.wdl == ((i % 3) - 1))
            r.record(f"at({i}).features",
                     list(pos.features) == [1, 2, 3, 4, 5, 6, 7, 8])
    finally:
        try: os.remove(path)
        except OSError: pass


def test_bulk_views_are_zerocopy(r):
    section("Bulk array accessors (zero-copy NumPy views)")
    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name

    try:
        make_sample(path)
        rdr = cnnp.Reader(path)

        flags = rdr.flags_array()
        evals = rdr.eval_array()
        wdls  = rdr.wdl_array()
        wflat = rdr.w_flat()
        anchors = rdr.block_anchors()
        prefix  = rdr.block_prefix()

        r.record("flags.dtype == uint8",   flags.dtype == np.uint8)
        r.record("evals.dtype == int16",   evals.dtype == np.int16)
        r.record("wdls.dtype  == int8",    wdls.dtype  == np.int8)
        r.record("wflat.dtype == uint16",  wflat.dtype == np.uint16)
        r.record("anchors.dtype == uint64", anchors.dtype == np.uint64)
        r.record("prefix.dtype  == uint16", prefix.dtype  == np.uint16)

        r.record("flags.shape == (4,)",  flags.shape == (4,))
        r.record("evals.shape == (4,)",  evals.shape == (4,))
        r.record("wdls.shape  == (4,)",  wdls.shape  == (4,))
        r.record("wflat.shape == (32,)", wflat.shape == (32,))   # 4 pos * 8 features
        r.record("anchors.shape == (1,)", anchors.shape == (1,))
        r.record("prefix.shape == (1025,)", prefix.shape == (1025,))

        # Zero-copy check: array.base should be the Reader (or hold reference to it)
        r.record("flags array has base (zero-copy)", flags.base is not None)
        r.record("wflat array has base (zero-copy)", wflat.base is not None)

        # WDL values should match what we wrote: (i % 3) - 1
        expected_wdls = [(i % 3) - 1 for i in range(4)]
        r.record("wdl values match", list(wdls) == expected_wdls)

        # w_flat should be the features pattern repeated 4 times
        expected_wflat = [1, 2, 3, 4, 5, 6, 7, 8] * 4
        r.record("w_flat pattern", list(wflat) == expected_wflat)
    finally:
        try: os.remove(path)
        except OSError: pass


def test_metadata_is_json(r):
    section("Metadata trailer (JSON)")
    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name

    try:
        make_sample(path)
        rdr = cnnp.Reader(path)
        md = rdr.metadata
        r.record("metadata is bytes", isinstance(md, bytes))
        r.record("metadata is non-empty", len(md) > 0)

        parsed = json.loads(md.decode("utf-8"))
        r.record("metadata parses as JSON", isinstance(parsed, dict))
        r.record("metadata.format == cnnp_sparse_v2",
                 parsed.get("format") == "cnnp_sparse_v2")
        r.record("metadata.layout == single_file",
                 parsed.get("layout") == "single_file")
    finally:
        try: os.remove(path)
        except OSError: pass


def test_header_dict(r):
    section("Reader.header (dict)")
    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name

    try:
        make_sample(path)
        rdr = cnnp.Reader(path)
        h = rdr.header
        r.record("header is dict",                    isinstance(h, dict))
        r.record("header.version == 2",               h["version"] == 2)
        r.record("header.feature_set == HalfP",       h["feature_set"] == cnnp.FeatureSet.HalfP)
        r.record("header.num_positions == 4",         h["num_positions"] == 4)
        r.record("header.num_features_total == 32",   h["num_features_total"] == 32)
        r.record("header.flags_offset == 256",        h["flags_offset"] == 256)
    finally:
        try: os.remove(path)
        except OSError: pass


def test_validation_errors(r):
    section("Error handling")

    # 1) Missing file → MmapError
    try:
        cnnp.Reader("__nonexistent_xyz123.cnnp")
        r.record("missing file raises MmapError", False, "no exception")
    except cnnp.MmapError:
        r.record("missing file raises MmapError", True)
    except Exception as e:
        r.record("missing file raises MmapError", False, f"got {type(e).__name__}")

    # 2) Bad stm in Writer → WriteError
    try:
        w = cnnp.Writer()
        feats = np.array([1, 2], dtype=np.uint16)
        w.add(score_cp=0, wdl=0, stm=2, piece_count=2, features=feats)
        r.record("bad stm raises WriteError", False, "no exception")
    except cnnp.WriteError:
        r.record("bad stm raises WriteError", True)
    except Exception as e:
        r.record("bad stm raises WriteError", False, f"got {type(e).__name__}")

    # 3) Empty Writer.finalize → WriteError
    try:
        w = cnnp.Writer()
        with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
            tp = f.name
        try:
            w.finalize(tp)
            r.record("empty finalize raises WriteError", False, "no exception")
        finally:
            try: os.remove(tp)
            except OSError: pass
    except cnnp.WriteError:
        r.record("empty finalize raises WriteError", True)


def test_at_out_of_range(r):
    section("Out-of-range access")
    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name
    try:
        make_sample(path)
        rdr = cnnp.Reader(path)
        try:
            rdr.at(rdr.num_positions)
            r.record("at(N) raises", False, "no exception")
        except Exception as e:
            r.record("at(N) raises", True, f"({type(e).__name__})")
    finally:
        try: os.remove(path)
        except OSError: pass


def test_max_in_memory_bytes_guardrail(r):
    section("max_in_memory_bytes guardrail")
    # Cap at 1024 bytes — way less than one block's prefix (2050B).
    w = cnnp.Writer(max_in_memory_bytes=1024)
    feats = np.array([1, 2], dtype=np.uint16)
    w.add(score_cp=0, wdl=0, stm=0, piece_count=2, features=feats)

    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name
    try:
        try:
            w.finalize(path)
            r.record("max_in_memory_bytes triggers WriteError", False, "no exception")
        except cnnp.WriteError:
            r.record("max_in_memory_bytes triggers WriteError", True)
        except Exception as e:
            r.record("max_in_memory_bytes triggers WriteError", False,
                     f"got {type(e).__name__}")
    finally:
        try: os.remove(path)
        except OSError: pass


def test_module_validate_helper(r):
    section("Module-level validate()")
    with tempfile.NamedTemporaryFile(suffix=".cnnp", delete=False) as f:
        path = f.name
    try:
        make_sample(path)
        try:
            cnnp.validate(path)
            r.record("validate(good_file) returns None", True)
        except Exception as e:
            r.record("validate(good_file) returns None", False, str(e))

        # Corrupt the magic
        with open(path, "r+b") as f:
            f.write(b"XXXX")
        try:
            cnnp.validate(path)
            r.record("validate(corrupt) raises", False)
        except cnnp.ParseError:
            r.record("validate(corrupt) raises", True)
        except Exception as e:
            r.record("validate(corrupt) raises", False, type(e).__name__)
    finally:
        try: os.remove(path)
        except OSError: pass


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Running cnnp Python smoke tests")
    print(f"  cnnp module:  {cnnp.__file__}")
    print(f"  cnnp version: {cnnp.__version__}")
    print(f"  numpy:        {np.__version__}")

    r = TestResult()
    try:
        test_module_constants(r)
        test_writer_reader_roundtrip(r)
        test_bulk_views_are_zerocopy(r)
        test_metadata_is_json(r)
        test_header_dict(r)
        test_validation_errors(r)
        test_at_out_of_range(r)
        test_max_in_memory_bytes_guardrail(r)
        test_module_validate_helper(r)
    except Exception:
        traceback.print_exc()
        return 1

    print(f"\n-- Summary --")
    print(f"  passed: {r.passed}")
    print(f"  failed: {r.failed}")
    if r.failed:
        print("\nFailures:")
        for name, msg in r.failures:
            print(f"  - {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
