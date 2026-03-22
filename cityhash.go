package catboost

import (
	"encoding/binary"
	"math/bits"
)

// CatBoost uses a fork of CityHash 1.0 that produces different results than
// the mainline Google CityHash. This is a port of their implementation from:
// https://github.com/catboost/catboost/blob/master/util/digest/city.cpp

const (
	k0 uint64 = 0xc3a5c85c97cb3127
	k1 uint64 = 0xb492b66fbe98f273
	k2 uint64 = 0x9ae16a3b2f90404f
	k3 uint64 = 0xc949d7c7509e6557
)

var le = binary.LittleEndian

func rotateByAtLeast1(val uint64, shift int) uint64 {
	return bits.RotateLeft64(val, -shift)
}

func shiftMix(val uint64) uint64 {
	return val ^ (val >> 47)
}

func hash128to64(low, high uint64) uint64 {
	const mul uint64 = 0x9ddfea08eb382d69
	a := (low ^ high) * mul
	a ^= (a >> 47)
	b := (high ^ a) * mul
	b ^= (b >> 47)
	b *= mul
	return b
}

func hashLen16(u, v uint64) uint64 {
	return hash128to64(u, v)
}

func hashLen0to16(s []byte) uint64 {
	length := len(s)

	if length > 8 {
		a := le.Uint64(s[0:8])
		b := le.Uint64(s[length-8 : length])
		return hashLen16(a, rotateByAtLeast1(b+uint64(length), length)) ^ b
	}

	if length >= 4 {
		a := uint64(le.Uint32(s[0:4]))
		return hashLen16(uint64(length)+(a<<3), uint64(le.Uint32(s[length-4:length])))
	}

	if length > 0 {
		a := uint32(s[0])
		b := uint32(s[length>>1])
		c := uint32(s[length-1])
		y := a + (b << 8)
		z := uint32(length) + (c << 2)
		return shiftMix(uint64(y)*k2^uint64(z)*k3) * k2
	}

	return k2
}

func hashLen17to32(s []byte) uint64 {
	length := len(s)
	a := le.Uint64(s[0:8]) * k1
	b := le.Uint64(s[8:16])
	c := le.Uint64(s[length-8:length]) * k2
	d := le.Uint64(s[length-16:length-8]) * k0
	return hashLen16(
		bits.RotateLeft64(a-b, -43)+bits.RotateLeft64(c, -30)+d,
		a+bits.RotateLeft64(b^k3, -20)-c+uint64(length),
	)
}

func hashLen33to64(s []byte) uint64 {
	length := len(s)
	z := le.Uint64(s[24:32])
	a := le.Uint64(s[0:8]) + (uint64(length)+le.Uint64(s[length-16:length-8]))*k0
	b := bits.RotateLeft64(a+z, -52)
	c := bits.RotateLeft64(a, -37)
	a += le.Uint64(s[8:16])
	c += bits.RotateLeft64(a, -7)
	a += le.Uint64(s[16:24])
	vf := a + z
	vs := b + bits.RotateLeft64(a, -31) + c

	a = le.Uint64(s[16:24]) + le.Uint64(s[length-32:length-24])
	z = le.Uint64(s[length-8 : length])
	b = bits.RotateLeft64(a+z, -52)
	c = bits.RotateLeft64(a, -37)
	a += le.Uint64(s[length-24 : length-16])
	c += bits.RotateLeft64(a, -7)
	a += le.Uint64(s[length-16 : length-8])
	wf := a + z
	ws := b + bits.RotateLeft64(a, -31) + c

	r := shiftMix((vf+ws)*k2 + (wf+vs)*k0)
	return shiftMix(r*k0+vs) * k2
}

type uint64Pair struct {
	first, second uint64
}

func weakHashLen32WithSeeds(w, x, y, z, a, b uint64) uint64Pair {
	a += w
	b = bits.RotateLeft64(b+a+z, -21)
	c := a
	a += x
	a += y
	b += bits.RotateLeft64(a, -44)
	return uint64Pair{a + z, b + c}
}

func weakHashLen32WithSeedsBytes(s []byte, a, b uint64) uint64Pair {
	return weakHashLen32WithSeeds(
		le.Uint64(s[0:8]),
		le.Uint64(s[8:16]),
		le.Uint64(s[16:24]),
		le.Uint64(s[24:32]),
		a,
		b,
	)
}

func cityHash64(s []byte) uint64 {
	length := len(s)

	if length <= 16 {
		return hashLen0to16(s)
	} else if length <= 32 {
		return hashLen17to32(s)
	} else if length <= 64 {
		return hashLen33to64(s)
	}

	// For strings > 64 bytes.
	x := le.Uint64(s[0:8])
	y := le.Uint64(s[length-16:length-8]) ^ k1
	z := le.Uint64(s[length-56:length-48]) ^ k0
	v := weakHashLen32WithSeedsBytes(s[length-64:length-32], uint64(length), y)
	w := weakHashLen32WithSeedsBytes(s[length-32:length], uint64(length)*k1, k0)
	z += shiftMix(v.second) * k1
	x = bits.RotateLeft64(z+x, -39) * k1
	y = bits.RotateLeft64(y, -33) * k1

	length = (length - 1) & ^int(63)
	for length > 0 {
		x = bits.RotateLeft64(x+y+v.first+le.Uint64(s[16:24]), -37) * k1
		y = bits.RotateLeft64(y+v.second+le.Uint64(s[48:56]), -42) * k1
		x ^= w.second
		y ^= v.first
		z = bits.RotateLeft64(z^w.first, -33)
		v = weakHashLen32WithSeedsBytes(s[0:32], v.second*k1, x+w.first)
		w = weakHashLen32WithSeedsBytes(s[32:64], z+w.second, y)
		z, x = x, z
		s = s[64:]
		length -= 64
	}

	return hashLen16(
		hashLen16(v.first, w.first)+shiftMix(y)*k1+z,
		hashLen16(v.second, w.second)+x,
	)
}
