package gloom

// Snapshots for sharded filters.
//
// ShardedAtomicFilter deliberately has no MarshalBinary, for the reason given on [AtomicFilter]:
// an Add sets k bits across k separate words without a lock, so serializing a filter that still
// has writers can capture part of an item and produce a filter that returns false negatives.
// That reasoning is about taking bits OUT of a live filter, and it is correct. This file adds a
// format for the two operations that reasoning does not cover:
//
//   - [ShardedAtomicFilter.WriteSnapshot] takes bits out, and therefore requires the caller to
//     guarantee the filter is quiescent. The precondition is the caller's to enforce; it cannot
//     be checked here.
//   - [ShardedAtomicFilter.MergeSnapshot] puts bits in. Bloom filters only ever set bits, so a
//     union cannot damage an item that is already present, and merging is safe while the filter
//     is being written to.
//
// The asymmetry is the point: removing bits is unsafe, adding bits is not, and only the latter
// can be offered without a precondition.
//
// # Format
//
// The header is little-endian:
//
//   - Version (1 byte): snapshot format version
//   - NumShards (8 bytes): number of shards
//   - Count (8 bytes): number of items added
//   - NumBlocks (8 bytes * NumShards): per-shard number of 512-bit blocks
//   - K (4 bytes * NumShards): per-shard number of hash functions
//
// followed by each shard's block words (NumBlocks * 64 bytes each) and a CRC-32C over that
// shard's bytes (4 bytes).
//
// The geometry is recorded rather than recomputed because a key's bit position depends on all
// of it: the shard comes from the high half of the 128-bit hash, and the block index is reduced
// within that shard. A filter with a different shard count, per-shard block count, or k hashes
// the same key to a different place, and there is no way to detect that after the fact, because
// such a filter still answers every Test correctly by its own geometry. Loading therefore
// refuses a snapshot whose geometry does not match the destination.
//
// Both directions stream through a fixed-size window. A sharded filter sized for a billion
// items is gigabytes, so nothing here buffers a whole shard or a whole payload.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

const (
	// shardedSnapshotVersion identifies the on-disk layout. It is separate from
	// serializeVersion so the single-filter and sharded formats can change independently.
	shardedSnapshotVersion byte = 1

	// shardedMaxShards bounds the shard count accepted from a snapshot. ShardedAtomicFilter
	// rounds shard counts up to a power of two and real deployments use roughly one per core,
	// so this is far above any real value while still preventing a corrupt header from driving
	// a large allocation.
	shardedMaxShards = uint64(1) << 20
)

var (
	// ErrSnapshotGeometry is returned when a snapshot does not describe the destination
	// filter, and loading it would leave every key hashing to the wrong shard or block.
	ErrSnapshotGeometry = errors.New("gloom: sharded snapshot geometry does not match filter")

	// ErrSnapshotCorrupt is returned when a snapshot fails to parse or fails a checksum.
	ErrSnapshotCorrupt = errors.New("gloom: corrupt sharded snapshot")
)

type shardedSnapshotHeader struct {
	Version   byte
	NumShards uint64
	Count     uint64
	NumBlocks []uint64
	K         []uint32
}

// validate checks that the header is internally consistent before any of it is used to size an
// allocation or copy into a filter.
func (h *shardedSnapshotHeader) validate() error {
	if h.Version != shardedSnapshotVersion {
		return fmt.Errorf("%w: unsupported version %d", ErrSnapshotCorrupt, h.Version)
	}
	if h.NumShards == 0 || h.NumShards > shardedMaxShards {
		return fmt.Errorf("%w: shard count %d out of range", ErrSnapshotCorrupt, h.NumShards)
	}
	if uint64(len(h.NumBlocks)) != h.NumShards || uint64(len(h.K)) != h.NumShards {
		return fmt.Errorf("%w: header shard arrays do not match shard count", ErrSnapshotCorrupt)
	}
	for i := range h.NumBlocks {
		if h.NumBlocks[i] == 0 || h.NumBlocks[i] > maxNumBlocks {
			return fmt.Errorf("%w: shard %d block count %d out of range", ErrSnapshotCorrupt, i, h.NumBlocks[i])
		}
		if GetPrimePartition(h.K[i]) == nil {
			return fmt.Errorf("%w: shard %d has unsupported k=%d", ErrSnapshotCorrupt, i, h.K[i])
		}
	}
	return nil
}

// WriteSnapshot serializes the filter to w.
//
// The filter MUST be quiescent for the entire call: no goroutine may call Add or AddString on
// it. This is a precondition that cannot be checked at runtime. A write that races this call
// can have only some of its k bits copied, and the resulting snapshot will report an item as
// absent that was never removed. A bloom filter that returns a false negative is broken, and
// no amount of validation on load can recover from it, because the missing bit is
// indistinguishable from an item that was never in the filter. If writers cannot be stopped,
// build a separate filter from the same source data instead of snapshotting a live one.
//
// Reading is safe concurrently: Test may run against the same filter while it is serialized,
// because a reader cannot introduce a torn item.
//
// Each shard is checksummed separately so that a reader can reject a damaged shard without
// first consuming the entire payload.
func (f *ShardedAtomicFilter) WriteSnapshot(w io.Writer) error {
	bw := bufio.NewWriterSize(w, 1<<20)

	header := make([]byte, 1+8+8+8*f.numShards+4*f.numShards)
	header[0] = shardedSnapshotVersion
	binary.LittleEndian.PutUint64(header[1:9], f.numShards)
	binary.LittleEndian.PutUint64(header[9:17], f.Count())

	off := 17
	for _, shard := range f.shards {
		binary.LittleEndian.PutUint64(header[off:off+8], shard.numBlocks)
		off += 8
	}
	for _, shard := range f.shards {
		binary.LittleEndian.PutUint32(header[off:off+4], shard.k)
		off += 4
	}
	if _, err := bw.Write(header); err != nil {
		return fmt.Errorf("writing sharded snapshot header: %w", err)
	}

	// Blocks go out through a fixed-size window for the same reason the reader uses one: a
	// shard is large enough that buffering one to checksum it would add a copy of the filter's
	// own memory at snapshot time.
	const chunk = 1 << 20
	buf := make([]byte, min(chunk, BlockWords*8))
	hash := crc32.New(crc32cTable)
	for i, shard := range f.shards {
		hash.Reset()
		remaining := int64(shard.numBlocks) * BlockWords * 8
		var wordBase int64
		for remaining > 0 {
			n := min(remaining, int64(len(buf)))
			window := buf[:n]
			for off := int64(0); off+8 <= n; off += 8 {
				binary.LittleEndian.PutUint64(window[off:off+8], shard.blocks[wordBase+off/8].Load())
			}
			wordBase += n / 8
			remaining -= n
			if _, err := hash.Write(window); err != nil {
				return fmt.Errorf("checksumming shard %d: %w", i, err)
			}
			if _, err := bw.Write(window); err != nil {
				return fmt.Errorf("writing shard %d: %w", i, err)
			}
		}
		var sum [4]byte
		binary.LittleEndian.PutUint32(sum[:], hash.Sum32())
		if _, err := bw.Write(sum[:]); err != nil {
			return fmt.Errorf("writing shard %d checksum: %w", i, err)
		}
	}

	if err := bw.Flush(); err != nil {
		return fmt.Errorf("flushing sharded snapshot: %w", err)
	}
	return nil
}

// MergeSnapshot reads a snapshot from r and merges it into f.
//
// Unlike [ShardedAtomicFilter.WriteSnapshot], this is safe to call while the filter is being
// written to, and needs no lock. The snapshot is unioned into f word by word rather than
// replacing it. Bloom filters set bits and never clear them, so a union cannot damage a
// present item:
//
//   - every item in the snapshot is present afterwards,
//   - every item added concurrently is present afterwards,
//   - no interleaving of the two can invalidate either,
//   - merging the same snapshot twice is idempotent.
//
// f is left unmodified on failure. The header and geometry are validated before anything is
// merged, and each shard's checksum is verified before that shard's bits are applied, so a
// caller that gets an error can fall back to filling the filter another way without first
// discarding what it already has.
//
// The item count advances by the snapshot's count so that EstimatedFalsePositiveRate remains
// meaningful. Items present in both are counted twice, matching what Count does when the same
// key is added twice through Add.
func (f *ShardedAtomicFilter) MergeSnapshot(r io.Reader) error {
	br := bufio.NewReaderSize(r, 1<<20)

	head := make([]byte, 17)
	if _, err := io.ReadFull(br, head); err != nil {
		return fmt.Errorf("%w: reading header: %w", ErrSnapshotCorrupt, err)
	}
	h := shardedSnapshotHeader{
		Version:   head[0],
		NumShards: binary.LittleEndian.Uint64(head[1:9]),
		Count:     binary.LittleEndian.Uint64(head[9:17]),
	}
	if h.NumShards == 0 || h.NumShards > shardedMaxShards {
		return fmt.Errorf("%w: shard count %d out of range", ErrSnapshotCorrupt, h.NumShards)
	}
	h.NumBlocks = make([]uint64, h.NumShards)
	h.K = make([]uint32, h.NumShards)

	perShardBlocks := make([]byte, 8*h.NumShards)
	if _, err := io.ReadFull(br, perShardBlocks); err != nil {
		return fmt.Errorf("%w: reading shard block counts: %w", ErrSnapshotCorrupt, err)
	}
	for i := range h.NumBlocks {
		h.NumBlocks[i] = binary.LittleEndian.Uint64(perShardBlocks[i*8 : i*8+8])
	}
	perShardK := make([]byte, 4*h.NumShards)
	if _, err := io.ReadFull(br, perShardK); err != nil {
		return fmt.Errorf("%w: reading shard k values: %w", ErrSnapshotCorrupt, err)
	}
	for i := range h.K {
		h.K[i] = binary.LittleEndian.Uint32(perShardK[i*4 : i*4+4])
	}
	if err := h.validate(); err != nil {
		return err
	}

	if h.NumShards != f.numShards {
		return fmt.Errorf("%w: snapshot has %d shards, filter has %d", ErrSnapshotGeometry, h.NumShards, f.numShards)
	}
	for i, shard := range f.shards {
		if h.NumBlocks[i] != shard.numBlocks || h.K[i] != shard.k {
			return fmt.Errorf("%w: shard %d is %d blocks k=%d, filter is %d blocks k=%d",
				ErrSnapshotGeometry, i, h.NumBlocks[i], h.K[i], shard.numBlocks, shard.k)
		}
	}

	// A fixed-size window keeps memory flat while blocks are copied into the filter.
	const chunk = 1 << 20
	scratch := make([]byte, min(chunk, BlockWords*8))
	hash := crc32.New(crc32cTable)
	for i, shard := range f.shards {
		hash.Reset()
		remaining := int64(shard.numBlocks) * BlockWords * 8
		// wordBase tracks the destination word, which advances across windows.
		var wordBase int64
		for remaining > 0 {
			n := min(remaining, int64(len(scratch)))
			buf := scratch[:n]
			if _, err := io.ReadFull(br, buf); err != nil {
				return fmt.Errorf("%w: reading shard %d: %w", ErrSnapshotCorrupt, i, err)
			}
			if _, err := hash.Write(buf); err != nil {
				return fmt.Errorf("%w: checksumming shard %d: %w", ErrSnapshotCorrupt, i, err)
			}
			for off := int64(0); off+8 <= n; off += 8 {
				shard.blocks[wordBase+off/8].Or(binary.LittleEndian.Uint64(buf[off : off+8]))
			}
			wordBase += n / 8
			remaining -= n
		}
		var sum [4]byte
		if _, err := io.ReadFull(br, sum[:]); err != nil {
			return fmt.Errorf("%w: reading shard %d checksum: %w", ErrSnapshotCorrupt, i, err)
		}
		if got := hash.Sum32(); got != binary.LittleEndian.Uint32(sum[:]) {
			return fmt.Errorf("%w: shard %d checksum mismatch", ErrSnapshotCorrupt, i)
		}
	}

	// The snapshot's item count is applied in one step rather than once per item, which
	// matters when a filter holds hundreds of millions of them. Count sums across shards, so
	// charging the whole snapshot to one shard reports the right total.
	f.shards[0].count.addN(h.Count)
	return nil
}

// addN adds n to the counter through a single stripe. It exists for snapshot loads, which add
// items in bulk rather than one at a time.
func (c *stripedCounter) addN(n uint64) {
	if n == 0 {
		return
	}
	c.stripes[0].v.Add(n)
}
