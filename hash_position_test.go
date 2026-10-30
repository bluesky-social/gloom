package gloom

import "testing"

// keyPosition is where a key lands in a filter of a fixed shape.
type keyPosition struct {
	shard     uint64
	blockIdx  uint64
	intraHash uint32
}

// positionOf computes where a key lands for the fixed geometry used below: 8 shards and
// 1000 blocks per shard. The block count is deliberately not a power of two so that
// reduceRange's multiply-shift reduction is exercised on a value where a modulo-based or
// otherwise differently-reduced implementation would disagree.
func positionOf(s string) keyPosition {
	h := hashRawString128(s)
	blockIdx, intra := hashSplitSharded(h, 1000)
	return keyPosition{shard: shardIndexFromHash(h, 7), blockIdx: blockIdx, intraHash: intra}
}

// TestKeyPositionIsStable pins where a handful of keys land. The expectations are literal
// rather than computed from the functions under test, because a test that derives its
// expectations from the same code it checks passes no matter what that code does.
//
// If this fails, a change altered the key-to-position mapping, so every serialized snapshot
// means something different to this build than to the one that wrote it, and loading one
// yields false negatives across the whole keyspace. Bump ShardedSnapshotFormat (see hash.go)
// and update these values in the same commit; updating the numbers alone leaves other builds
// free to load a snapshot they will misread.
func TestKeyPositionIsStable(t *testing.T) {
	tests := []struct {
		key  string
		want keyPosition
	}{
		{"", keyPosition{shard: 0, blockIdx: 600, intraHash: 0x268c8a5b}},
		{"a", keyPosition{shard: 4, blockIdx: 661, intraHash: 0xf8507ca9}},
		{"hello", keyPosition{shard: 7, blockIdx: 710, intraHash: 0x992bf7b2}},
		{"attractor", keyPosition{shard: 5, blockIdx: 773, intraHash: 0x5b94f5d6}},
		{"the quick brown fox", keyPosition{shard: 1, blockIdx: 156, intraHash: 0xd5a3411d}},
		{"\x00", keyPosition{shard: 2, blockIdx: 651, intraHash: 0xc305332f}},
		{"ünïcodé", keyPosition{shard: 3, blockIdx: 779, intraHash: 0x731dbdfa}},
	}

	for _, tt := range tests {
		got := positionOf(tt.key)
		if got != tt.want {
			t.Errorf("key %q lands at shard=%d block=%d intra=%#x, want shard=%d block=%d intra=%#x: "+
				"the key-to-position mapping changed. Bump ShardedSnapshotFormat (see hash.go) "+
				"and update this test in the same commit.",
				tt.key, got.shard, got.blockIdx, got.intraHash, tt.want.shard, tt.want.blockIdx, tt.want.intraHash)
		}
	}
}

// TestKeyHashingMatchesAcrossEntryPoints checks that a []byte key and the equivalent string
// key hash identically. Filters are written through whichever entry point the caller has and
// read through the same key in the other form after a round trip through storage, so if these
// two ever diverge, a snapshot built one way silently misses every key queried the other way.
func TestKeyHashingMatchesAcrossEntryPoints(t *testing.T) {
	const s = "the quick brown fox"
	hs := hashRawString128(s)
	hb := hashRaw128([]byte(s))
	if hs != hb {
		t.Fatalf("string and []byte keys hash %q differently (%v vs %v)", s, hs, hb)
	}
}
