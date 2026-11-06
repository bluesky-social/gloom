package gloom

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"strconv"
	"sync"
	"testing"
)

// newShardedForTest builds a small sharded filter with a fixed shard count so tests do not
// depend on GOMAXPROCS.
func newShardedForTest() *ShardedAtomicFilter {
	return NewShardedAtomic(1_000, 0.001, 4)
}

func TestShardedSnapshotRoundtrip(t *testing.T) {
	src := newShardedForTest()
	items := make([]string, 5_000)
	for i := range items {
		items[i] = "actor" + strconv.Itoa(i) + "|subject" + strconv.Itoa(i)
		src.AddString(items[i])
	}

	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}

	dst := newShardedForTest()
	if err := dst.MergeSnapshot(&buf); err != nil {
		t.Fatalf("MergeSnapshot: %v", err)
	}

	// A snapshot is only useful if it carries every item: a missing key is indistinguishable
	// from one that was never added, so a caller that treats a false result as proof of absence
	// would act on a wrong answer.
	for _, item := range items {
		if !dst.TestString(item) {
			t.Fatalf("false negative after snapshot load for %q", item)
		}
	}
	if got, want := dst.Count(), src.Count(); got != want {
		t.Fatalf("count after load = %d, want %d", got, want)
	}
}

// TestShardedSnapshotMergeIsIdempotent covers loading the same snapshot twice, which happens
// on any retry and must not corrupt the filter or inflate the bit array.
func TestShardedSnapshotMergeIsIdempotent(t *testing.T) {
	src := newShardedForTest()
	for i := range 500 {
		src.AddString("a" + strconv.Itoa(i))
	}
	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}
	data := bytes.Clone(buf.Bytes())

	dst := newShardedForTest()
	if err := dst.MergeSnapshot(bytes.NewReader(data)); err != nil {
		t.Fatalf("first MergeSnapshot: %v", err)
	}
	fill := dst.EstimatedFillRatio()
	if err := dst.MergeSnapshot(bytes.NewReader(data)); err != nil {
		t.Fatalf("second MergeSnapshot: %v", err)
	}
	if got := dst.EstimatedFillRatio(); got != fill {
		t.Fatalf("fill ratio changed on re-merge: %v -> %v", fill, got)
	}
	for i := range 500 {
		if !dst.TestString("a" + strconv.Itoa(i)) {
			t.Fatalf("item %d lost on re-merge", i)
		}
	}
}

// TestShardedSnapshotMergesConcurrentlyWithAdds is the reason MergeSnapshot unions instead of
// replacing: a live process merges a snapshot while its own writers keep adding. Both sets of
// items must survive, and the load must not be a data race.
func TestShardedSnapshotMergesConcurrentlyWithAdds(t *testing.T) {
	src := newShardedForTest()
	snapshotItems := make([]string, 2_000)
	for i := range snapshotItems {
		snapshotItems[i] = "snapshot|" + strconv.Itoa(i)
		src.AddString(snapshotItems[i])
	}

	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}

	dst := newShardedForTest()

	const liveItems = 2_000
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range liveItems {
			dst.AddString("live|" + strconv.Itoa(i))
		}
	}()

	if err := dst.MergeSnapshot(&buf); err != nil {
		t.Fatalf("MergeSnapshot: %v", err)
	}
	wg.Wait()

	for _, item := range snapshotItems {
		if !dst.TestString(item) {
			t.Fatalf("snapshot item %q lost during concurrent merge", item)
		}
	}
	for i := range liveItems {
		if !dst.TestString("live|" + strconv.Itoa(i)) {
			t.Fatalf("live item %d lost during concurrent merge", i)
		}
	}
}

func TestShardedSnapshotRejectsGeometryMismatch(t *testing.T) {
	src := NewShardedAtomic(1_000, 0.001, 4)
	src.AddString("a|b")
	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}

	// Different shard count. Loading this would hash every key to the wrong shard, so it must
	// be refused rather than silently accepted.
	dst := NewShardedAtomic(1_000, 0.001, 8)
	if err := dst.MergeSnapshot(bytes.NewReader(buf.Bytes())); !errors.Is(err, ErrSnapshotGeometry) {
		t.Fatalf("MergeSnapshot error = %v, want ErrSnapshotGeometry", err)
	}

	// Same shard count, different per-shard block count and therefore different block indices.
	dst2 := NewShardedAtomic(1_000_000, 0.001, 4)
	if err := dst2.MergeSnapshot(bytes.NewReader(buf.Bytes())); !errors.Is(err, ErrSnapshotGeometry) {
		t.Fatalf("MergeSnapshot error = %v, want ErrSnapshotGeometry", err)
	}

	// Same size but a different target false positive rate also changes the geometry.
	dst3 := NewShardedAtomic(1_000, 0.00001, 4)
	if err := dst3.MergeSnapshot(bytes.NewReader(buf.Bytes())); !errors.Is(err, ErrSnapshotGeometry) {
		t.Fatalf("MergeSnapshot error = %v, want ErrSnapshotGeometry", err)
	}
}

// TestShardedSnapshotGeometryMismatchTouchesNothing verifies the refusal happens before any
// bits are merged, so a caller can fall back and use the filter as if the load never ran.
func TestShardedSnapshotGeometryMismatchTouchesNothing(t *testing.T) {
	src := NewShardedAtomic(1_000, 0.001, 4)
	src.AddString("a|b")
	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}

	dst := NewShardedAtomic(1_000, 0.001, 8)
	if err := dst.MergeSnapshot(bytes.NewReader(buf.Bytes())); !errors.Is(err, ErrSnapshotGeometry) {
		t.Fatalf("MergeSnapshot error = %v, want ErrSnapshotGeometry", err)
	}
	if got := dst.Count(); got != 0 {
		t.Fatalf("count = %d after rejected load, want 0", got)
	}
	if got := dst.EstimatedFillRatio(); got != 0 {
		t.Fatalf("fill ratio = %v after rejected load, want 0", got)
	}
}

func TestShardedSnapshotRejectsCorruption(t *testing.T) {
	src := newShardedForTest()
	for i := range 100 {
		src.AddString(strconv.Itoa(i))
	}
	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}

	tests := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"truncated header", buf.Bytes()[:10]},
		{"truncated payload", buf.Bytes()[:len(buf.Bytes())/2]},
		{"flipped bit in bit array", flipByte(buf.Bytes(), len(buf.Bytes())/2)},
		{"flipped bit in checksum", flipByte(buf.Bytes(), len(buf.Bytes())-1)},
		{"bad version", setByte(buf.Bytes(), 0, 99)},
		{"zero shard count", zeroUint64(buf.Bytes(), 1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := newShardedForTest()
			err := dst.MergeSnapshot(bytes.NewReader(tt.data))
			if !errors.Is(err, ErrSnapshotCorrupt) {
				t.Fatalf("MergeSnapshot error = %v, want ErrSnapshotCorrupt", err)
			}
		})
	}
}

// TestShardedSnapshotCorruptShardIsNotApplied covers the case where the damaged shard comes
// first. Bits cannot be cleared, so a shard merged before its checksum is verified stays in the
// filter permanently, and the caller's scan fallback only adds true positives on top of it. That
// would leave an over-full shard for the life of the process while the caller believes it fell
// back cleanly. Nothing from a shard may be applied until that shard is verified.
func TestShardedSnapshotCorruptShardIsNotApplied(t *testing.T) {
	const numShards = 4
	src := NewShardedAtomic(1_000, 0.001, numShards)
	items := make([]string, 5_000)
	for i := range items {
		items[i] = "actor" + strconv.Itoa(i) + "|subject" + strconv.Itoa(i)
		src.AddString(items[i])
	}

	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}

	// Damaging a byte in shard 0's blocks must fail its checksum. Shard 0 sits immediately
	// after the header, so corrupting a byte there cannot touch any other shard.
	headerLen := 1 + 8 + 8 + 8*numShards + 4*numShards
	data := flipByte(buf.Bytes(), headerLen+16)

	dst := NewShardedAtomic(1_000, 0.001, numShards)
	if err := dst.MergeSnapshot(bytes.NewReader(data)); !errors.Is(err, ErrSnapshotCorrupt) {
		t.Fatalf("MergeSnapshot error = %v, want ErrSnapshotCorrupt", err)
	}

	// A failed load must leave the filter as it was: no bits from the unverified shard, and in
	// this case nothing at all, since the damaged shard is the first one read.
	for _, item := range items {
		if dst.TestString(item) {
			t.Fatalf("%q present after a failed load: bits from an unverified shard were applied", item)
		}
	}
	if got := dst.Count(); got != 0 {
		t.Fatalf("count after failed load = %d, want 0", got)
	}
}

// TestShardedSnapshotSurvivesStreamingBoundaries drives a filter large enough that both the
// writer and reader cross their fixed-size windows, since the offset arithmetic for partial
// windows only shows up at that size.
func TestShardedSnapshotSurvivesStreamingBoundaries(t *testing.T) {
	src := NewShardedAtomic(4_000_000, 0.0001, 2)
	items := make([]string, 20_000)
	for i := range items {
		items[i] = "boundary" + strconv.Itoa(i)
		src.AddString(items[i])
	}

	var buf bytes.Buffer
	if err := src.WriteSnapshot(&buf); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}
	if buf.Len() <= 1<<20 {
		t.Fatalf("payload is only %d bytes; the test did not cross the window size", buf.Len())
	}

	dst := NewShardedAtomic(4_000_000, 0.0001, 2)
	if err := dst.MergeSnapshot(&buf); err != nil {
		t.Fatalf("MergeSnapshot: %v", err)
	}
	for _, item := range items {
		if !dst.TestString(item) {
			t.Fatalf("false negative after streaming load for %q", item)
		}
	}
}

// TestSnapshotSize checks that the accessor reports the exact encoded length for both writers and
// that the length depends only on the filter's geometry, not on how many items it holds.
func TestSnapshotSize(t *testing.T) {
	src := NewShardedAtomic(10_000, 0.001, 4)

	var empty bytes.Buffer
	if err := src.WriteSnapshot(&empty); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}
	if got, want := int64(empty.Len()), src.SnapshotSize(); got != want {
		t.Fatalf("WriteSnapshot wrote %d bytes, SnapshotSize reports %d", got, want)
	}

	for i := range 1_000 {
		src.AddString("size|" + strconv.Itoa(i))
	}
	var full bytes.Buffer
	if err := src.WriteSnapshotWithReplay(&full); err != nil {
		t.Fatalf("WriteSnapshotWithReplay: %v", err)
	}
	if got, want := int64(full.Len()), src.SnapshotSize(); got != want {
		t.Fatalf("WriteSnapshotWithReplay wrote %d bytes, SnapshotSize reports %d", got, want)
	}
	if empty.Len() != full.Len() {
		t.Fatalf("encoded length changed with item count: %d -> %d", empty.Len(), full.Len())
	}
}

// TestShardedSnapshotEncodingIsStable pins the encoded bytes of a small fixture: the header field
// order and widths, each shard's block words in shard order, and the CRC-32C trailer after each
// shard. It also proves a quiescent filter encodes identically through both public writers, so the
// replay-aware path cannot drift from the standalone one.
func TestShardedSnapshotEncodingIsStable(t *testing.T) {
	src := NewShardedAtomic(200, 0.01, 2)
	for _, item := range []string{"alpha|1", "beta|2", "gamma|3"} {
		src.AddString(item)
	}

	var quiet bytes.Buffer
	if err := src.WriteSnapshot(&quiet); err != nil {
		t.Fatalf("WriteSnapshot: %v", err)
	}
	var replay bytes.Buffer
	if err := src.WriteSnapshotWithReplay(&replay); err != nil {
		t.Fatalf("WriteSnapshotWithReplay: %v", err)
	}
	if !bytes.Equal(quiet.Bytes(), replay.Bytes()) {
		t.Fatalf("quiescent filter encoded differently through the two writers")
	}

	// Assemble the expected bytes from the documented layout without going through the encoder,
	// so a change to field order, width, or checksum placement fails here.
	numShards := src.NumShards()
	blocksPerShard := src.NumBlocks() / numShards
	want := make([]byte, 0, src.SnapshotSize())
	want = append(want, ShardedSnapshotFormat)
	want = binary.LittleEndian.AppendUint64(want, numShards)
	want = binary.LittleEndian.AppendUint64(want, src.Count())
	for range numShards {
		want = binary.LittleEndian.AppendUint64(want, blocksPerShard)
	}
	for range numShards {
		want = binary.LittleEndian.AppendUint32(want, src.K())
	}
	for _, shard := range src.shards {
		payloadStart := len(want)
		for word := range shard.blocks {
			want = binary.LittleEndian.AppendUint64(want, shard.blocks[word].Load())
		}
		sum := crc32.Checksum(want[payloadStart:], crc32cTable)
		want = binary.LittleEndian.AppendUint32(want, sum)
	}

	if !bytes.Equal(quiet.Bytes(), want) {
		t.Fatalf("encoded snapshot does not match the documented layout")
	}
	if got, size := int64(len(want)), src.SnapshotSize(); got != size {
		t.Fatalf("laid-out length %d disagrees with SnapshotSize %d", got, size)
	}
}

// TestSnapshotWithReplayPreservesCompletedAdds covers the guarantee the replay-aware writer makes:
// every addition that completed before the call began is in the output, even though new additions
// keep landing while it serializes. Items added after the call starts may be absent, which is
// exactly why the caller replays from its own boundary.
func TestSnapshotWithReplayPreservesCompletedAdds(t *testing.T) {
	const items = 5_000
	src := NewShardedAtomic(4_000_000, 0.0001, 2)
	for i := range items {
		src.AddString("before|" + strconv.Itoa(i))
	}

	// Block once roughly one flush has gone out, so the write is genuinely mid-serialization
	// and not merely waiting to start.
	writer := newBlockingWriter(1 << 20)
	writeDone := make(chan error, 1)
	go func() { writeDone <- src.WriteSnapshotWithReplay(writer) }()

	select {
	case <-writer.blocked:
	case err := <-writeDone:
		t.Fatalf("snapshot completed without blocking: %v", err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 2_000 {
			src.AddString("live|" + strconv.Itoa(i))
		}
	}()
	wg.Wait()
	close(writer.release)

	if err := <-writeDone; err != nil {
		t.Fatalf("WriteSnapshotWithReplay: %v", err)
	}

	dst := NewShardedAtomic(4_000_000, 0.0001, 2)
	if err := dst.MergeSnapshot(bytes.NewReader(writer.bytes())); err != nil {
		t.Fatalf("MergeSnapshot: %v", err)
	}
	for i := range items {
		if !dst.TestString("before|" + strconv.Itoa(i)) {
			t.Fatalf("addition %d completed before the write but is absent from the snapshot", i)
		}
	}
}

// blockingWriter collects a snapshot and, once limit bytes have been written, parks until release
// is closed. A test uses it to open a window in which additions land while serialization is
// already underway.
type blockingWriter struct {
	limit   int
	blocked chan struct{}
	release chan struct{}

	pause sync.Once
	buf   bytes.Buffer
}

func newBlockingWriter(limit int) *blockingWriter {
	return &blockingWriter{
		limit:   limit,
		blocked: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (w *blockingWriter) Write(p []byte) (int, error) {
	if w.buf.Len() >= w.limit {
		w.pause.Do(func() { close(w.blocked) })
		<-w.release
	}
	return w.buf.Write(p)
}

// bytes returns the collected snapshot. It is only valid after the writer has returned.
func (w *blockingWriter) bytes() []byte {
	return w.buf.Bytes()
}

func flipByte(src []byte, idx int) []byte {
	out := bytes.Clone(src)
	out[idx] ^= 0xff
	return out
}

func setByte(src []byte, idx int, v byte) []byte {
	out := bytes.Clone(src)
	out[idx] = v
	return out
}

// zeroUint64 clears the little-endian uint64 at idx, used to zero a header field.
func zeroUint64(src []byte, idx int) []byte {
	out := bytes.Clone(src)
	for i := range 8 {
		out[idx+i] = 0
	}
	return out
}
