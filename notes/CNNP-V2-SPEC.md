# CNNP V2 — Format Specification

> **Chess-NN-Pack (CNNP)** is a binary format for Stockfish-style NNUE
> training data. V2 replaces HDF5 sparse_v1 with a flat, mmap-first,
> zero-dependency layout designed specifically for chess training
> pipelines.

**Status: FROZEN.** This document is the authoritative wire-format and
implementation contract for CNNP V2. Any future changes require a new
version (`magic = "CNN3"` or bumped `version_u16`) and a separate spec
document. The implementation may begin against this spec.

---

## 1. Goals

1. **mmap-first random access** — O(1) per position, no B-tree, no chunks
2. **Zero external dependencies** in the reader — just mmap + struct unpack
3. **~25% smaller** than equivalent HDF5 sparse_v1 (without bit-packing)
4. **Self-describing** via embedded JSON metadata trailer
5. **Forward-compatible** with sharded V3 (header fields already reserved)
6. **Auditable** — single fixed-size header, predictable layout

## 2. Non-goals

- Compression (mmap is the priority; compression breaks zero-copy)
- Multi-writer concurrency
- Cross-cluster distribution
- Transactional semantics
- General scientific data interchange (HDF5 wins there)

---

## 3. File structure

```
+---------------------------------------------------+
|  HEADER (256 bytes, fixed)                        |
+---------------------------------------------------+
|  flags_u8         [num_positions]                 |
|  eval_i16         [num_positions]                 |
|  wdl_i8           [num_positions]                 |
|  block_anchors_u64[num_blocks]                    |
|  block_prefix_u16 [num_blocks * (block_size+1)]   |
|  w_flat_u16       [num_features_total]            |
+---------------------------------------------------+
|  METADATA (JSON, length = metadata_length)        |
+---------------------------------------------------+
|  reserved padding (optional, e.g. checksum)       |
+---------------------------------------------------+
```

All multi-byte values are **little-endian**. Float types are IEEE 754
binary32.

---

## 4. Header (256 bytes)

| Off | Size | Field                  | Type    | Value (V2)  | Notes |
|-----|------|------------------------|---------|-------------|-------|
| 0   | 4    | `magic`                | `[u8;4]`| `"CNN2"`    | ASCII identifier |
| 4   | 2    | `version`              | u16     | `2`         | Format version |
| 6   | 2    | `header_size`          | u16     | `256`       | Always 256 in V2 |
| 8   | 1    | `endian`               | u8      | `0`         | 0 = little-endian (only) |
| 9   | 1    | `layout_kind`          | u8      | `0`         | 0 = single_file |
| 10  | 1    | `feature_set`          | u8      |             | 1=HalfP, 2=HalfKAv2_hm, 3=HalfKA |
| 11  | 1    | `count_semantics`      | u8      | `0`         | 0 = piece_count_equals_nnz |
| 12  | 1    | `eval_encoding`        | u8      | `1`         | 1 = int16_fixed_normalized |
| 13  | 1    | `mate_encoding`        | u8      | `0`         | 0 = in_band, 1 = saturate, 2 = out_band (V3) |
| 14  | 1    | `flags_encoding`       | u8      | `0`         | 0 = stm + count_minus_2 layout |
| 15  | 1    | `reserved`             | u8      | `0`         | Must be 0 |
| 16  | 4    | `shard_id`             | u32     | `0`         | V2: always 0 |
| 20  | 4    | `num_shards`           | u32     | `1`         | V2: always 1 |
| 24  | 8    | `global_position_start`| u64     | `0`         | V2: always 0 |
| 32  | 8    | `num_positions`        | u64     |             | Total positions in this file |
| 40  | 8    | `num_features_total`   | u64     |             | sum of nnz across all positions |
| 48  | 4    | `max_feature_id`       | u32     |             | **Exclusive upper bound** of feature ids. Valid ids are `0 <= id < max_feature_id`. For HalfP: `768` → ids `0..767`. |
| 52  | 4    | `num_blocks`           | u32     |             | `ceil(num_positions / block_size)` |
| 56  | 2    | `block_size`           | u16     | `1024`      | Always 1024 in V2 |
| 58  | 1    | `max_count`            | u8      | `32`        | Max nnz per position |
| 59  | 1    | `count_base`           | u8      | `2`         | `piece_count = stored + count_base` |
| 60  | 4    | `fixed_scale`          | f32     | `3000.0`    | eval_i16 = round(target * fixed_scale) |
| 64  | 4    | `normal_eval_clip`     | f32     | `6.5`       | Documentation only |
| 68  | 4    | `storage_target_clip`  | f32     | `10.0`      | Storage range bound |
| 72  | 1    | `has_eval_global`      | u8      | `1`         | All positions have eval |
| 73  | 1    | `has_wdl_global`       | u8      | `1`         | All positions have WDL |
| 74  | 2    | `header_flags`         | u16     | `0`         | Reserved bitfield |
| 76  | 4    | `reserved`             | u32     | `0`         | Must be 0 |
| 80  | 8    | `flags_offset`         | u64     |             | byte offset of flags array |
| 88  | 8    | `eval_offset`          | u64     |             | byte offset of eval array |
| 96  | 8    | `wdl_offset`           | u64     |             | byte offset of wdl array |
| 104 | 8    | `block_anchors_offset` | u64     |             | byte offset of block_anchors |
| 112 | 8    | `block_prefix_offset`  | u64     |             | byte offset of block_prefix |
| 120 | 8    | `w_flat_offset`        | u64     |             | byte offset of w_flat |
| 128 | 8    | `metadata_offset`      | u64     |             | byte offset of JSON trailer |
| 136 | 8    | `metadata_length`      | u64     |             | byte length of JSON trailer |
| 144 | 112  | `reserved padding`     | `[u8;112]` | all 0    | Future use |

All `u64` fields are 8-byte aligned. All `u32` fields are 4-byte aligned.
All `u16` fields are 2-byte aligned.

---

## 5. Encoding rules

### 5.1 `flags_u8` — per-position packed scalars

```text
bit 0     stm                          0=white, 1=black
bits 1-5  count_minus_2                0..30 (sentinel 31 reserved)
bits 6-7  reserved                     must be 0
```

**Encode (C++20, validating):**
```cpp
inline std::uint8_t encode_flags(std::uint8_t stm, std::uint8_t piece_count) {
    // Validate inputs BEFORE arithmetic (avoid unsigned underflow on piece_count - 2).
    if (stm > 1)                                throw EncodeError("invalid stm");
    if (piece_count < 2 || piece_count > 32)    throw EncodeError("invalid piece_count");

    const std::uint8_t stored_count =
        static_cast<std::uint8_t>(piece_count - 2);    // safe: 0..30
    return static_cast<std::uint8_t>((stored_count << 1) | (stm & 0x01));
}
```

**Decode (C++20):**
```cpp
const std::uint8_t stm           = flags & 0x01;
const std::uint8_t stored_count  = (flags >> 1) & 0x1F;   // 0..31
const std::uint8_t piece_count   = stored_count + 2;       // count_base = 2
// validate: stored_count != 31  (sentinel reserved)
```

### 5.2 `eval_i16` — normalized fixed-point

The encoder first runs the standard centipawn → normalized-target
conversion (matching the Stockfish-style encoding):

```cpp
inline float normalize_target(std::int32_t score_cp) {
    // Use int64 to avoid UB on std::abs(INT32_MIN).
    const std::int64_t cp     = static_cast<std::int64_t>(score_cp);
    const std::int64_t abs_cp = (cp < 0) ? -cp : cp;
    const float        sign   = (cp >= 0) ? 1.0f : -1.0f;

    if (abs_cp > 25'000) {
        // Mate encoding: above normal_eval_clip range, distinguishable.
        const float ply_estimate = std::clamp(
            static_cast<float>(32'000 - abs_cp) / 100.0f, 1.0f, 30.0f);
        return sign * (10.0f - ply_estimate * 0.1f);     // range ≈ [7.0, 9.9]
    }
    return sign * std::min(static_cast<float>(abs_cp) / 400.0f, 6.5f);   // [-6.5, 6.5]
}
```

The actual mate-target range produced by this function is approximately
`[7.0, 9.9]` (because `ply_estimate` is clamped to `[1, 30]`, never 0).
`storage_target_clip = 10.0` is the broader allowed bound, providing
margin without saturating any value the encoder produces.

Then stored as i16:

```cpp
float target = normalize_target(score_cp);
target = std::clamp(target, -10.0f, 10.0f);              // storage_target_clip
const std::int16_t eval_i16 =
    static_cast<std::int16_t>(std::lroundf(target * 3000.0f));   // fixed_scale
```

**Decode:**
```cpp
const float target_normalized = static_cast<float>(eval_i16) / 3000.0f;
```

**Range check:**
- `10.0 * 3000 = 30000` → fits in i16 (max 32767)
- `-10.0 * 3000 = -30000` → fits in i16 (min -32768)
- Resolution: `1 / 3000 ≈ 0.000333` (negligible loss vs fp32 for NNUE)
- **Mate scores are preserved without saturation** (`mate_encoding = "in_band"`)

### 5.3 `wdl_i8` — game outcome, white-POV

```text
-1   black won (white lost)
 0   draw
+1   white won
```

The binpack stores result from STM perspective; this format normalizes
to white-POV at conversion time.

### 5.4 HalfP feature indices (in `w_flat`)

```text
feature_idx = color * 384 + piece_type * 64 + square
  color:       0=white, 1=black     (the piece color; the board is white-POV)
  piece_type:  0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
  square:      0..63 (a1=0, h8=63)
```

Range: `0..767` (`max_feature_id = 768`). Stored as `u16`.

### 5.5 Offsets — anchored block-prefix scheme

For each block of `block_size` (= 1024) positions:

- `block_anchors_u64[b]` = absolute **feature index** into `w_flat` at
  the start of block `b` (i.e. an element index into the `u16` array,
  **not** a byte offset). The byte offset of `w_flat` itself is in the
  header field `w_flat_offset`.
- `block_prefix_u16[b * 1025 + j]` = local prefix sum of `nnz` within
  block `b` for positions `0..j`. Always starts at 0 and ends at the
  block's total feature count.

**Random access for position `i` (C++20, safe API):**

```cpp
if (i >= num_positions)        throw OutOfRange("position");
const std::uint64_t b           = i / block_size;
const std::uint64_t j           = i % block_size;
const std::uint64_t prefix_base = b * (block_size + 1);
const std::uint64_t start =
    block_anchors[b] + static_cast<std::uint64_t>(block_prefix[prefix_base + j]);
const std::uint64_t end =
    block_anchors[b] + static_cast<std::uint64_t>(block_prefix[prefix_base + j + 1]);
if (end < start || end > num_features_total) throw Corruption("feature slice");

std::span<const std::uint16_t> features{w_flat + start, end - start};
// w_flat is a `const std::uint16_t*` aimed at byte offset `w_flat_offset`
// inside the mmap region (zero-copy view).
```

**Padding rule for the last block.** Because `block_prefix_u16` always
contains `num_blocks * (block_size + 1)` entries, the final block has
unused slots whenever `num_positions % block_size != 0`. The encoder
**MUST** fill those unused slots by repeating the last valid prefix
value. This guarantees:

- The file is bit-deterministic for the same input
- The validator can check `prefix[j+1] >= prefix[j]` uniformly
- The reader can index `block_prefix[prefix_base + j + 1]` without
  branching on edge cases

**Properties:**

- **O(1) random access** (no scan, no B-tree)
- `block_size * max_count = 1024 * 32 = 32_768` fits in `u16` (max 65535) ✓
- For 2B positions: ~4 GB total prefix overhead (vs ~16 GB for plain
  `i64` offsets in HDF5)

---

## 6. Metadata trailer (JSON)

A UTF-8 JSON object at byte offset `metadata_offset`, length
`metadata_length`. Free-form attributes for forward compatibility.

**Recommended schema:**

```json
{
  "format": "cnnp_sparse_v2",
  "layout": "single_file",
  "shards": [
    {
      "path": "positions.cnnp",
      "shard_id": 0,
      "global_position_start": 0,
      "num_positions": 2000000000
    }
  ],
  "source": {
    "binpack": "farseerT76.binpack",
    "binpack_size_bytes": 6141321549,
    "binpack_sha256": "..."
  },
  "conversion": {
    "tool": "binpack_to_cnnp v0.1.0",
    "started_at": "2026-05-12T10:00:00Z",
    "duration_seconds": 487.3,
    "num_threads": 16
  },
  "filter": {
    "skip_captures": true,
    "skip_in_check": true,
    "wld_filtered": false,
    "filter_pct": 23.6
  },
  "encoding_params": {
    "eval_scale_cp": 400.0,
    "normal_eval_clip": 6.5,
    "storage_target_clip": 10.0,
    "fixed_scale": 3000,
    "mate_threshold_cp": 25000,
    "mate_base": 10.0,
    "mate_decay": 0.1
  },
  "stats": {
    "wdl_white_pct": 33.3,
    "wdl_draw_pct": 39.4,
    "wdl_loss_pct": 27.3,
    "sign_agreement_pct": 99.0
  }
}
```

The JSON object MAY contain additional fields. Readers MUST ignore
unknown fields. Required fields for V2:

- `format` (must equal `"cnnp_sparse_v2"`)
- `layout` (must equal `"single_file"` in V2)

---

## 7. Validator invariants

A spec-compliant V2 file MUST satisfy all of the following:

### Header

```
magic == "CNN2"
version == 2
header_size == 256
endian == 0
layout_kind == 0
feature_set ∈ {1, 2, 3}
block_size == 1024
max_count == 32
count_base == 2
count_semantics == 0
eval_encoding == 1
mate_encoding ∈ {0, 1}              (V2: 0 expected; 1 allowed)
flags_encoding == 0
has_eval_global == 1                 (V2: arrays are always present)
has_wdl_global  == 1                 (V2: arrays are always present)
shard_id == 0, num_shards == 1, global_position_start == 0
all reserved bytes == 0
```

### Consistency

```
num_positions > 0                                  (V2: empty datasets forbidden)
num_blocks    > 0                                  (V2: at least one block required)
num_blocks == ceil(num_positions / block_size)
ceil(num_positions / block_size) ≤ UINT32_MAX      (num_blocks fits u32)
storage_target_clip * fixed_scale ≤ 32767
block_size * max_count ≤ 65535                     (prefix u16 invariant)
all *_offset within [header_size, file_size)
metadata_offset + metadata_length ≤ file_size
data offsets monotonic and non-overlapping
```

### Array sizes match declared offsets

Each array must occupy exactly the bytes its element count + element
size requires. The recommended canonical layout (encoder convention,
not strictly required by the format) is:

```
flags_offset         + num_positions * 1            == eval_offset
eval_offset          + num_positions * 2            == wdl_offset
wdl_offset           + num_positions * 1            == block_anchors_offset
block_anchors_offset + num_blocks    * 8            == block_prefix_offset
block_prefix_offset  + num_blocks    * (block_size + 1) * 2  == w_flat_offset
w_flat_offset        + num_features_total * 2       == metadata_offset
```

A spec-compliant validator MUST at minimum verify that **each array
range is large enough** to hold its declared element count:

```
flags_offset         + num_positions * 1  ≤ eval_offset
eval_offset          + num_positions * 2  ≤ wdl_offset
wdl_offset           + num_positions * 1  ≤ block_anchors_offset
block_anchors_offset + num_blocks * 8     ≤ block_prefix_offset
block_prefix_offset  + num_blocks * (block_size + 1) * 2  ≤ w_flat_offset
w_flat_offset        + num_features_total * 2  ≤ metadata_offset
```

(All multiplications above are subject to the **checked arithmetic**
requirement.) Files that follow the canonical layout exactly are
strongly preferred for predictability.

**Checked arithmetic.** All computations involving `num_positions`,
`num_features_total`, `num_blocks`, or per-array byte sizes MUST be
performed in `u64` with overflow detection (e.g.,
`__builtin_mul_overflow` or equivalent). A spec-compliant validator
rejects files whose declared sizes overflow when multiplied by the
element width.

### Alignment of typed array offsets

The reader constructs typed `const T*` views into the mmap region;
those pointers must be properly aligned for type `T`:

```
flags_offset         : any byte alignment (u8)
eval_offset          : eval_offset         % 2 == 0    (i16)
wdl_offset           : any byte alignment (i8)
block_anchors_offset : block_anchors_offset % 8 == 0   (u64)
block_prefix_offset  : block_prefix_offset  % 2 == 0   (u16)
w_flat_offset        : w_flat_offset        % 2 == 0   (u16)
metadata_offset      : any byte alignment (opaque bytes)
```

Note: by the layout in §4, all `u64` fields in the header itself are
also 8-byte aligned, so a writer that follows the layout naturally
satisfies these alignment constraints.

### Per-position arrays

```
flags_u8[i]:
  ((flags >> 1) & 0x1F) ≠ 31         (sentinel reserved)
  ((flags >> 6) & 0x03) == 0         (reserved bits zero)

wdl_i8[i] ∈ {-1, 0, +1}

eval_i16[i] ∈ [-storage_target_clip * fixed_scale,
               +storage_target_clip * fixed_scale]
```

### Per-block prefix arrays

```
block_anchors[0] == 0
block_anchors[b+1] == block_anchors[b] + block_prefix[b][block_size]
                                         (for 0 ≤ b < num_blocks - 1)

block_prefix[b][0] == 0
block_prefix[b][j+1] ≥ block_prefix[b][j]                 (monotonic)
block_prefix[b][block_size] ≤ block_size * max_count

For the LAST block (b == num_blocks - 1) when num_positions is not a
multiple of block_size: unused slots beyond the last valid position
MUST be filled with the last valid prefix value (padding rule from §5.5).
```

### Global feature count

`block_prefix[b][j]` represents the **count of feature elements**
contributed by the first `j` positions of block `b` (so `prefix[0] = 0`
and `prefix[N]` = total features for `N` positions). The number of
valid positions in the final block is:

```
valid_positions_in_last_block =
    (num_positions % block_size == 0)
        ? block_size
        : num_positions % block_size
```

Then:

```
all w_flat[k] < max_feature_id

block_anchors[num_blocks - 1]
    + block_prefix[last_block][valid_positions_in_last_block]
    == num_features_total
```

(Note: `valid_positions_in_last_block` is a count, so the prefix
index used is `valid_positions_in_last_block`, not
`valid_positions_in_last_block - 1`.)

---

## 8. Estimated size (2 billion positions, HalfP, avg 20 features)

| Section          | Bytes per position | 2B positions   |
|------------------|--------------------|----------------|
| `flags_u8`       | 1                  | 2.0 GB         |
| `eval_i16`       | 2                  | 4.0 GB         |
| `wdl_i8`         | 1                  | 2.0 GB         |
| `block_anchors`  | `8 / 1024`         | ~16 MB         |
| `block_prefix`   | `2 * 1025 / 1024`  | ~4.0 GB        |
| `w_flat`         | `40` (20 × u16)    | ~80.0 GB       |
| header + metadata|                    | ~1 KB          |
| **Total**        |                    | **~92 GB**     |
| HDF5 sparse_v1   |                    | ~119 GB        |
| **Reduction**    |                    | **~23%**       |

---

## 9. Design decisions log

### 9.1 Why `fixed_scale = 3000` (not 4096)?

The pipeline produces normalized targets in `[-6.5, 6.5]` for normal
positions and `[-10.0, 10.0]` for mate positions. With `fixed_scale = 4096`:

```
i16 max / 4096 = 32767 / 4096 ≈ 7.9998
```

Mate scores above ~8.0 would saturate, destroying the distinction
between mate-in-1 (target ≈ 9.97) and mate-in-30 (target ≈ 7.0).

`fixed_scale = 3000`:

```
10.0 * 3000 = 30000   (fits in i16)
1 / 3000 ≈ 0.000333   (resolution loss vs fp32 is irrelevant for NNUE)
```

Not a power of 2, but the decode is `eval_i16 / 3000.0` (one fp
operation either way), so no measurable performance difference.

### 9.2 Why `count_minus_2` (not raw piece_count)?

`piece_count` ranges from 2 (two kings only) to 32 (initial position) —
that's 31 distinct values. 5 bits store 32 distinct values (0..31), so
storing the raw value would consume the full range with no sentinel.

By storing `piece_count - 2`:
- range 0..30 fits in 5 bits
- value 31 is reserved as an invalid sentinel
- `count_base = 2` field documents the offset

This frees the sentinel for future use and gives the validator a
cheap corruption check.

### 9.3 Why `num_features_total` is `u64` (not `u32`)?

For 2B positions × ~20 features = ~40B features, which exceeds `u32`
max (~4.3B). Catching this at the spec level avoids silent overflow.

### 9.4 Why explicit array offsets in the header?

Each array could be placed at a computed offset from the previous
array's size. Instead, V2 stores 7 explicit `u64` offsets (56 bytes).

Reasons:
- Reader doesn't need to compute or understand the layout dependencies
- Future versions can reorder, remove, or pad arrays without breaking
  V2 readers (as long as they still read what they expect)
- Cost is negligible (56 bytes in a multi-GB file)

### 9.5 Why JSON metadata trailer (not binary)?

- **Inspectable:** `tail -c $LEN file.cnnp | jq .`
- **Forward-compatible:** new fields are ignored by old readers
- **Tiny overhead:** typically a few KB out of multi-GB files
- **Standard tooling:** `jq`, Python `json`, every language has a parser

The fixed binary header carries only what the reader needs to mmap
correctly; everything else (filter info, conversion provenance,
statistics) goes in the JSON trailer.

### 9.6 Why single file in V2 (not sharded)?

Sharding is appealing for parallel writes and resume-on-failure, but
adds complexity:
- DataLoader must know shard boundaries
- Global random shuffle becomes shard-then-position
- More files to manage and validate

V2 keeps the single-file simplicity. The header **already reserves**
`shard_id`, `num_shards`, and `global_position_start` so V3 can switch
to sharding without breaking the V2 reader's mental model.

The spec recommends V2 readers treat the dataset abstractly as a list
of shards (with `num_shards = 1` in V2), so the V3 transition is a
no-op at the reader API level.

### 9.7 Why little-endian only?

Effectively all NNUE training happens on x86_64 or ARM64, both
little-endian. Big-endian readers must byte-swap when reading.
Documenting this constraint avoids dual-endian complexity in the spec.

### 9.8 Why `block_size = 1024`?

Constraint: `block_size * max_count ≤ u16 max = 65535` so that
`block_prefix_u16` can hold the largest possible prefix value.

With `max_count = 32`:
- `block_size = 1024` → max prefix value = 32768 ✓ (fits in u16)
- `block_size = 2048` → max prefix value = 65536 ✗ (overflows u16)

1024 is the largest power of two that satisfies the constraint, which
gives the fewest blocks (less anchor overhead).

### 9.9 Why `metadata_length` field (not implicit "rest of file")?

Without an explicit length, the JSON trailer would have to be the last
bytes of the file. This forecloses adding optional trailer sections
later (checksum, signature, etc).

Cost: 8 bytes. Benefit: forward compatibility for additional trailers.

### 9.10 Why `block_anchors` stores feature indices (not byte offsets)?

Two encodings were considered:

- **Byte offsets** into the mmap region (`block_anchors[b] = w_flat_offset + ...`)
- **Feature indices** into the typed `w_flat` array (`block_anchors[b] = N feature elements`)

Feature indices were chosen because:

- The reader holds a typed `const std::uint16_t*` aimed at
  `w_flat_offset`; arithmetic in element units composes cleanly with
  `block_prefix` (also in element units)
- Byte offsets would require dividing by `sizeof(u16)` in every random
  access, just to undo the multiplication done at encode time
- Element-index semantics make the validator invariant
  `block_anchors[b+1] == block_anchors[b] + block_prefix[b][last]`
  trivially true (no unit conversion)

The byte offset of `w_flat` itself is stored separately in the header
field `w_flat_offset`. The reader uses it to position its typed
pointer once at file open; from then on it works in element indices.

### 9.11 Why field-by-field header parsing (not `reinterpret_cast`)?

C++ structs do not have a guaranteed bit-exact layout match with a
documented wire format. Padding, member alignment, ABI variations, and
endianness make `reinterpret_cast<const Header*>(mmap_bytes)` a
portability landmine. Even with `__attribute__((packed))` or
`#pragma pack(1)`, unaligned `u64` access remains UB on some targets.

Field-by-field parsing through `read_uXX_le()` helpers is:
- Fully portable
- Only ~1× slower per header parse (tiny absolute cost: 256 bytes)
- Endian-explicit (no surprise on big-endian machines)
- Easier to extend — new fields don't require struct layout changes

The cost is one parse function and a small set of read helpers; the
benefit is zero UB and full portability.

---

## 10. Deferred to V3

These are explicitly out of scope for V2 but the format is designed to
accommodate them later without breaking changes:

- **Sharding** — header fields `shard_id`, `num_shards`,
  `global_position_start` are reserved for the transition
- **`w_flat` 10-bit packing** — would save ~30 GB but breaks zero-copy
  mmap; deferred until a benchmark proves the trade-off is worth it
- **`mate_encoding = "out_band"`** — dedicated bit + separate range for
  mate scores; preserves resolution for both normal evals and mates
- **Optional zstd compression** — header field `header_flags` reserves
  bits for indicating compressed sections
- **Multiple feature sets in one file** — e.g., HalfP and HalfKAv2_hm
  side-by-side for the same positions; would require schema changes

---

## 11. Comparison to alternatives

| Format            | Reader deps | Random access | mmap | Single file |
|-------------------|-------------|---------------|------|-------------|
| HDF5 sparse_v1    | libhdf5 (~10 MB) | O(log n) B-tree | partial (chunked) | yes |
| binpack stream    | sfbinpack       | sequential only  | no   | yes |
| Apache Arrow      | libarrow        | columnar (chunked) | yes | yes (or dir) |
| Parquet           | libparquet      | row-group based  | partial | yes (or dir) |
| LMDB              | liblmdb         | O(log n) B+tree  | yes  | yes |
| Zarr              | none required   | chunked          | yes  | many files |
| **CNNP V2**       | **none required** | **O(1) flat**  | **yes** | **yes** |

CNNP V2 trades general-purpose flexibility for chess-specific
specialization. Within its domain (NNUE training data), the
combination of zero reader dependencies, true O(1) random access, and
single-file mmap is unique.

---

## 12. C++ implementation notes

The reference implementation targets **C++20** with **CMake** as the build
system. The CNNP binary format is language-neutral; C++ is chosen for the
reference reader/writer because it integrates naturally with Stockfish,
nnue-pytorch, and existing chess-engine training pipelines.

### Build / dependency policy

- **Core library: zero external runtime dependencies.** Memory mapping is
  implemented directly via POSIX `mmap`/`munmap` on Unix-like systems and
  `CreateFileMapping`/`MapViewOfFile` on Windows.
- **Tests: `gtest` (optional)** — only the test binary depends on it.
- **Python bindings: `pybind11` (optional)** — only the Python wheel
  depends on it.
- **No `libhdf5`, `libarrow`, `fmt`, `libsfbinpack` in the core reader.**
  The whole point of CNNP is that a downstream consumer can vendor the
  reader as one or two small files with no system dependencies.

### C++20 feature subset

The implementation uses the well-supported subset of C++20:

```
USE:
  std::span                  (non-owning views over arrays)
  std::byte                  (typed raw bytes)
  std::optional, std::variant
  constexpr (everywhere it helps)
  structured bindings
  enum class
  static_assert
  std::filesystem            (CLI tooling only)
  std::endian                 (compile-time endianness checks)
  std::bit_cast               (ONLY for primitive type punning, e.g.
                               u32 ↔ f32 for IEEE float deserialization;
                               NEVER for whole struct ↔ bytes — see §12)

AVOID in V2:
  std::format                (compiler/library support uneven; use snprintf)
  modules                    (ecosystem still maturing)
  coroutines                 (no async I/O needed for mmap pipeline)
  std::ranges (heavy adapters) (cost vs benefit not obvious yet)
  reflection                 (C++23+, not portable)
```

### Project layout

```
chess-nn-pack/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── notes/                        (design docs — already exists)
│
├── include/
│   └── cnnp/
│       ├── cnnp.hpp              (umbrella header)
│       ├── header.hpp            (Header struct + parse/serialize)
│       ├── encoding.hpp          (encode/decode flags, eval, wdl)
│       ├── reader.hpp            (mmap-based Reader, PositionView)
│       ├── writer.hpp            (Writer class)
│       └── validator.hpp         (validation rules)
│
├── src/
│   ├── header.cpp
│   ├── encoding.cpp
│   ├── reader.cpp
│   ├── writer.cpp
│   ├── validator.cpp
│   └── cli/
│       └── cnnp_main.cpp         (dump/validate/inspect entry point)
│
├── tests/
│   ├── test_header.cpp
│   ├── test_encoding.cpp
│   ├── test_validator.cpp
│   ├── test_roundtrip.cpp
│   └── fixtures/
│       └── golden.cnnp           (small reference file checked in)
│
├── examples/
│   ├── convert_binpack.cpp       (sfbinpack-equivalent → cnnp::Writer)
│   ├── inspect.cpp               (cnnp::Reader stats dump)
│   └── pytorch_loader.py         (pybind11 binding example)
│
└── python/                       (optional, V2.1)
    ├── setup.py
    └── cnnp_py.cpp               (pybind11)
```

A single-include amalgamated `cnnp.hpp` may be generated later (à la
`stb_image.h`) for friction-free vendoring, but is **not** the V2
distribution model — V2 ships source + CMake.

### Platform mmap abstraction

Because `mmap` differs between POSIX and Windows, the reader keeps a
small `MMapRegion` class that wraps both:

```
Linux/macOS:  open() → fstat() → mmap() → munmap() → close()
Windows:      CreateFileW → CreateFileMappingW → MapViewOfFile
              → UnmapViewOfFile → CloseHandle (file + mapping)
```

This is the only platform-specific code in the core. Everything else
operates on `std::span<const std::byte>` views regardless of OS.

### Header parsing: explicit field-by-field, no `reinterpret_cast`

The header **MUST** be parsed field-by-field through endian-safe helper
functions. Reinterpreting raw bytes as a `Header` struct via
`reinterpret_cast` or `std::bit_cast` is **forbidden** because:

- C++ struct padding/alignment is implementation-defined; the in-memory
  layout of a hand-written `Header` struct is not guaranteed to match
  the wire layout in §4
- Unaligned access of `std::uint64_t` from an arbitrary mmap address is
  UB on some architectures (older ARM, MIPS)
- Endianness conversion still has to happen explicitly anyway

**Required pattern:**

```cpp
inline std::uint16_t read_u16_le(std::span<const std::byte> data,
                                 std::size_t off) {
    return static_cast<std::uint16_t>(data[off + 0]) |
          (static_cast<std::uint16_t>(data[off + 1]) << 8);
}

inline std::uint32_t read_u32_le(std::span<const std::byte> data,
                                 std::size_t off);
inline std::uint64_t read_u64_le(std::span<const std::byte> data,
                                 std::size_t off);
inline float         read_f32_le(std::span<const std::byte> data,
                                 std::size_t off);
```

The in-memory `Header` struct is a **logical** representation populated
from those reads. It does not have to match the on-disk layout
byte-for-byte; only the parser/serializer pair has to agree on the
wire format.

### Metadata: opaque bytes in the core

The fixed-size header is binary and parsed by the core. The JSON
metadata trailer (§6) is treated as **opaque UTF-8 bytes** by the core
reader — it exposes the trailer as `std::string_view` (or
`std::span<const std::byte>`) but does **not** parse JSON.

Reasons:
- Keeps the core dependency-free (C++ has no standard JSON parser)
- Forward-compatible: new metadata fields are inert to the core reader
- Full schema validation is performed by the CLI tooling (`cnnp validate`)
  or downstream code, which can pull in a JSON library if desired

Validation responsibilities are split:

| Layer | MUST | MAY |
|---|---|---|
| **Spec** (the file itself) | metadata bytes form a valid UTF-8 JSON object with required fields `format`, `layout` | carry additional fields beyond the spec |
| **Core reader** | validate `metadata_offset + metadata_length ≤ file_size` and expose the trailer as a `std::string_view` / `std::span<const std::byte>` | validate UTF-8 / parse JSON (allowed but not required) |
| **CLI validator / `cnnp validate`** | validate UTF-8, parse JSON, check required fields | report unknown fields informatively |

A spec-compliant file therefore always contains a valid UTF-8 JSON
trailer; the core reader simply chooses not to parse it, leaving full
schema validation to higher layers that may pull in a JSON library.

### Compiler support matrix (target)

| Compiler | Minimum version | Notes |
|---|---|---|
| GCC      | 10              | C++20 mostly supported; 11+ recommended |
| Clang    | 12              | Good C++20; libc++ 14+ for `std::span` features |
| MSVC     | 19.29 (VS 2019 16.10) | `/std:c++20`; tested on VS 2022 |
| Apple Clang | 14           | Xcode 14+ |

CI matrix: GCC 11 + Clang 14 + MSVC 19.30 on Linux/macOS/Windows x86_64.

### 64-bit process requirement

The reference implementation targets **64-bit processes only**. 32-bit
builds are explicitly unsupported because:

- `num_features_total` can exceed `UINT32_MAX` (e.g., 40 billion for
  2 billion HalfP positions × ~20 features)
- Files routinely exceed 4 GB; mmap address space and `std::size_t`
  must accommodate them
- `block_anchors_u64` is u64 by design; downcasting to a 32-bit
  `std::size_t` for indexing is unsafe

The reader MUST refuse to open a file when:

```cpp
if (num_features_total > std::numeric_limits<std::size_t>::max())
    throw OpenError("file requires 64-bit process");
```

In practice this branch is dead on every supported target, but the
check formalizes the requirement and protects against accidental
32-bit builds.

---

## 13. Implementation roadmap

The spec is final; implementation is the next phase. Suggested order:

1. **`cnnp/header`** — header struct, byte layout, parse/serialize,
   endian-safe loads, all `static_assert`s (~600 lines C++)
2. **`cnnp/encoding`** — encode/decode of `flags_u8`, `eval_i16`,
   `wdl_i8`, plus `normalize_target` (~250 lines)
3. **`cnnp/validator`** — all invariants from §7 (~400 lines)
4. **`cnnp/reader`** — `MMapRegion`, `Reader`, `PositionView`,
   `block_prefix`-based random access (~500 lines)
5. **`cnnp/writer`** — streaming writer + final header patch + JSON
   metadata trailer (~500 lines)
6. **`cnnp/cli`** — `cnnp dump`, `cnnp validate`, `cnnp inspect`
   (~300 lines)
7. **Tests (gtest)** — golden-file roundtrip, validator coverage,
   corruption tests (~500 lines)
8. **Optional: `python/`** — `pybind11` bindings exposing
   `Reader`/`PositionView` to PyTorch DataLoaders (~200 lines)

Total estimated: **~3000–3500 lines C++** for a complete V2
implementation including tests (Python bindings are extra).

---

## 14. References

- [Stockfish binpack](https://github.com/official-stockfish/Stockfish) — source format and filtering semantics
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) — reference filter implementation (`make_skip_predicate`)
- [HDF5 sparse_v1](https://www.hdfgroup.org/solutions/hdf5/) — schema layout inspiration (offsets + values)
- [AnnData (.h5ad)](https://anndata.readthedocs.io/) — CSR-on-disk pattern
- [Apache Arrow](https://arrow.apache.org/) — mmap-first design philosophy
