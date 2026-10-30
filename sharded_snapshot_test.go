package gloom

import (
	"bytes"
	"errors"
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

	// A snapshot is only useful if it carries every item: one missing key becomes a missing
	// block/mute check on a read path that treats a false result as proof of absence.
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
