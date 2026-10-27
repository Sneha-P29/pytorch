#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// VSX does not have native fp16 arithmetic instructions. All arithmetic ops
// widen to fp32 via xvcvhpsp (vec_extract_fp32_from_shorth/shortl), apply the
// VSX float op, then narrow back via xvcvsphp (vec_pack_to_short_fp32).
// Bitwise ops work directly on the int16 representation without conversion.
//
// Layout: 16 Half lanes stored across two 128-bit VMX registers:
//   _vec0 = lanes [0..7]  (vint16, low 128 bits)
//   _vec1 = lanes [8..15] (vint16, high 128 bits)
//
// For fp32 widening each vint16 register (8 x fp16) expands to two vfloat32
// registers (4 x fp32) using vec_extract_fp32_from_shortl/shorth, giving four
// fp32 sub-vectors: f0l, f0h, f1l, f1h (8 fp32 values total per half).

template <>
struct is_vec_specialized_for<c10::Half> : std::bool_constant<true> {};

template <>
class Vectorized<c10::Half> {
 private:
  union {
    struct {
      vint16 _vec0;
      vint16 _vec1;
    };
    struct {
      vbool16 _vecb0;
      vbool16 _vecb1;
    };
  } __attribute__((__may_alias__));

 public:
  // Widen one vint16 (8 fp16 lanes) to two vfloat32 (4+4 fp32 lanes).
  // The shorth/shortl names follow big-endian element numbering, so on
  // little-endian POWER shorth is taken first to keep widen + pack lane order.
  static C10_ALWAYS_INLINE void fp16_to_fp32(
      vint16 v,
      vfloat32& lo,
      vfloat32& hi) {
    // vec_extract_fp32_from_short* require vuint16 (unsigned), not vint16.
    vuint16 uv = reinterpret_cast<vuint16>(v);
    lo = vec_extract_fp32_from_shorth(uv);
    hi = vec_extract_fp32_from_shortl(uv);
  }

  // Narrow two vfloat32 (4+4 fp32 lanes) back to one vint16 (8 fp16 lanes).
  static C10_ALWAYS_INLINE vint16 fp32_to_fp16(vfloat32 lo, vfloat32 hi) {
    return (vint16)vec_pack_to_short_fp32(lo, hi);
  }

  // Apply a unary Vectorized<float> method via fp32 widening (for sleef ops).
  C10_ALWAYS_INLINE Vectorized<c10::Half> map_vec_float(
      Vectorized<float> (Vectorized<float>::*m)() const) const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    Vectorized<float> r0 = (Vectorized<float>(f0l, f0h).*m)();
    Vectorized<float> r1 = (Vectorized<float>(f1l, f1h).*m)();
    return {fp32_to_fp16(r0.vec0(), r0.vec1()), fp32_to_fp16(r1.vec0(), r1.vec1())};
  }

  // Apply a binary Vectorized<float> method via fp32 widening (for sleef ops).
  C10_ALWAYS_INLINE Vectorized<c10::Half> map2_vec_float(
      const Vectorized<c10::Half>& b,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&) const) const {
    vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
    fp16_to_fp32(_vec0, a0l, a0h);
    fp16_to_fp32(_vec1, a1l, a1h);
    fp16_to_fp32(b._vec0, b0l, b0h);
    fp16_to_fp32(b._vec1, b1l, b1h);
    Vectorized<float> r0 = (Vectorized<float>(a0l, a0h).*m)(Vectorized<float>(b0l, b0h));
    Vectorized<float> r1 = (Vectorized<float>(a1l, a1h).*m)(Vectorized<float>(b1l, b1h));
    return {fp32_to_fp16(r0.vec0(), r0.vec1()), fp32_to_fp16(r1.vec0(), r1.vec1())};
  }

  using value_type = c10::Half;
  using vec_internal_type = vint16;
  using vec_internal_mask_type = vbool16;
  using size_type = int;

  static constexpr size_type size() {
    return 16;
  }

  Vectorized() = default;

  C10_ALWAYS_INLINE Vectorized(vint16 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool16 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vint16 v0, vint16 v1) : _vec0{v0}, _vec1{v1} {}
  C10_ALWAYS_INLINE Vectorized(vbool16 v0, vbool16 v1)
      : _vecb0{v0}, _vecb1{v1} {}

  C10_ALWAYS_INLINE Vectorized(c10::Half scalar) {
    uint16_t bits;
    std::memcpy(&bits, &scalar, sizeof(bits));
    _vec0 = vec_splats((int16_t)bits);
    _vec1 = _vec0;
  }

  C10_ALWAYS_INLINE Vectorized(
      c10::Half v0,
      c10::Half v1,
      c10::Half v2,
      c10::Half v3,
      c10::Half v4,
      c10::Half v5,
      c10::Half v6,
      c10::Half v7,
      c10::Half v8,
      c10::Half v9,
      c10::Half v10,
      c10::Half v11,
      c10::Half v12,
      c10::Half v13,
      c10::Half v14,
      c10::Half v15) {
    _vec0 = vint16{
        (int16_t)v0.x,  (int16_t)v1.x,  (int16_t)v2.x,  (int16_t)v3.x,
        (int16_t)v4.x,  (int16_t)v5.x,  (int16_t)v6.x,  (int16_t)v7.x};
    _vec1 = vint16{
        (int16_t)v8.x,  (int16_t)v9.x,  (int16_t)v10.x, (int16_t)v11.x,
        (int16_t)v12.x, (int16_t)v13.x, (int16_t)v14.x, (int16_t)v15.x};
  }

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const { return _vec0; }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const { return _vec1; }

  // blend: select lanes from a or b based on compile-time mask.
  template <uint64_t mask>
  static std::enable_if_t<mask == 0, Vectorized<c10::Half>> C10_ALWAYS_INLINE
  blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    return a;
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask & 65535) == 65535, Vectorized<c10::Half>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    return b;
  }

  template <uint64_t mask>
  static std::enable_if_t<mask == 255, Vectorized<c10::Half>> C10_ALWAYS_INLINE
  blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    return {b._vec0, a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask > 0 && mask < 255), Vectorized<c10::Half>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    constexpr int16_t g0 = (mask & 1) * 0xffff;
    constexpr int16_t g1 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7 = ((mask & 128) >> 7) * 0xffff;
    const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};
    return {(vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st), a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && (mask & 65535) != 65535 && ((mask & 255) == 255)),
      Vectorized<c10::Half>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    constexpr int16_t g0_2 = (((mask >> 8) & 1)) * 0xffff;
    constexpr int16_t g1_2 = (((mask >> 8) & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = (((mask >> 8) & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = (((mask >> 8) & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = (((mask >> 8) & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = (((mask >> 8) & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = (((mask >> 8) & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = (((mask >> 8) & 128) >> 7) * 0xffff;
    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    return {b._vec0, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && (mask & 65535) != 65535 && ((mask & 255) == 0)),
      Vectorized<c10::Half>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    constexpr int16_t g0_2 = (((mask >> 8) & 1)) * 0xffff;
    constexpr int16_t g1_2 = (((mask >> 8) & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = (((mask >> 8) & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = (((mask >> 8) & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = (((mask >> 8) & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = (((mask >> 8) & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = (((mask >> 8) & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = (((mask >> 8) & 128) >> 7) * 0xffff;
    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    return {a._vec0, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && (mask & 65535) != 65535 && (mask & 255) != 0 &&
       (mask & 255) != 255),
      Vectorized<c10::Half>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
    constexpr int16_t g0 = (mask & 1) * 0xffff;
    constexpr int16_t g1 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7 = ((mask & 128) >> 7) * 0xffff;
    constexpr int16_t g0_2 = (((mask >> 8) & 1)) * 0xffff;
    constexpr int16_t g1_2 = (((mask >> 8) & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = (((mask >> 8) & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = (((mask >> 8) & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = (((mask >> 8) & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = (((mask >> 8) & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = (((mask >> 8) & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = (((mask >> 8) & 128) >> 7) * 0xffff;
    const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};
    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    return {
        (vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st),
        (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  static Vectorized<c10::Half> C10_ALWAYS_INLINE blendv(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      const Vectorized<c10::Half>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }

  template <typename step_t>
  static Vectorized<c10::Half> arange(
      c10::Half base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<c10::Half>(
        base,
        base + step,
        base + c10::Half(2) * step,
        base + c10::Half(3) * step,
        base + c10::Half(4) * step,
        base + c10::Half(5) * step,
        base + c10::Half(6) * step,
        base + c10::Half(7) * step,
        base + c10::Half(8) * step,
        base + c10::Half(9) * step,
        base + c10::Half(10) * step,
        base + c10::Half(11) * step,
        base + c10::Half(12) * step,
        base + c10::Half(13) * step,
        base + c10::Half(14) * step,
        base + c10::Half(15) * step);
  }

  static Vectorized<c10::Half> set(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      size_t count = size()) {
    switch (count) {
      case 0:  return a;
      case 1:  return blend<1>(a, b);
      case 2:  return blend<3>(a, b);
      case 3:  return blend<7>(a, b);
      case 4:  return blend<15>(a, b);
      case 5:  return blend<31>(a, b);
      case 6:  return blend<63>(a, b);
      case 7:  return blend<127>(a, b);
      case 8:  return blend<255>(a, b);
      case 9:  return blend<511>(a, b);
      case 10: return blend<1023>(a, b);
      case 11: return blend<2047>(a, b);
      case 12: return blend<4095>(a, b);
      case 13: return blend<8191>(a, b);
      case 14: return blend<16383>(a, b);
      case 15: return blend<32767>(a, b);
    }
    return b;
  }

  static Vectorized<c10::Half> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const int16_t*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const int16_t*>(ptr))};
    }
    __at_align__ c10::Half tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(c10::Half));
    return {
        vec_vsx_ld(offset0, reinterpret_cast<const int16_t*>(tmp_values)),
        vec_vsx_ld(offset16, reinterpret_cast<const int16_t*>(tmp_values))};
  }

  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<int16_t*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<int16_t*>(ptr));
    } else if (count > 0) {
      __at_align__ c10::Half tmp_values[size()];
      vec_vsx_st(_vec0, offset0, reinterpret_cast<int16_t*>(tmp_values));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<int16_t*>(tmp_values));
      std::memcpy(ptr, tmp_values, std::min(count, size()) * sizeof(c10::Half));
    }
  }

  const c10::Half& operator[](int idx) const = delete;
  c10::Half& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == c10::Half(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<c10::Half> isnan() const {
    // A fp16 value is NaN iff it is not equal to itself.
    auto ret = (*this == *this);
    return {vec_not(ret._vecb0), vec_not(ret._vecb1)};
  }

  bool has_inf_nan() const {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i]) || _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }

  Vectorized<c10::Half> angle() const {
    auto zero = Vectorized<c10::Half>(c10::Half(0));
    auto pi = Vectorized<c10::Half>(c10::Half(c10::pi<float>));
    auto tmp = blendv(zero, pi, *this < zero);
    return blendv(tmp, *this, isnan());
  }
  Vectorized<c10::Half> real() const { return *this; }
  Vectorized<c10::Half> imag() const { return Vectorized<c10::Half>(c10::Half(0)); }
  Vectorized<c10::Half> conj() const { return *this; }

  // abs/neg operate directly on fp16 bit patterns — no fp32 conversion needed.
  Vectorized<c10::Half> C10_ALWAYS_INLINE abs() const {
    // Clear the sign bit (bit 15) of each fp16 lane.
    const vint16 sign_mask = vec_splats((int16_t)0x8000);
    return {vec_andc(_vec0, sign_mask), vec_andc(_vec1, sign_mask)};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE neg() const {
    // Flip the sign bit (bit 15) of each fp16 lane.
    const vint16 sign_mask = vec_splats((int16_t)0x8000);
    return {vec_xor(_vec0, sign_mask), vec_xor(_vec1, sign_mask)};
  }

  // Non-sleef float ops: widen to fp32 with hardware instructions, apply VSX
  // op inline (GCC built-ins cannot be passed as function pointers), narrow.
  Vectorized<c10::Half> C10_ALWAYS_INLINE ceil() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    return {fp32_to_fp16(vec_ceil(f0l), vec_ceil(f0h)),
            fp32_to_fp16(vec_ceil(f1l), vec_ceil(f1h))};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE floor() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    return {fp32_to_fp16(vec_floor(f0l), vec_floor(f0h)),
            fp32_to_fp16(vec_floor(f1l), vec_floor(f1h))};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE round() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    return {fp32_to_fp16(vec_round(f0l), vec_round(f0h)),
            fp32_to_fp16(vec_round(f1l), vec_round(f1h))};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE trunc() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    return {fp32_to_fp16(vec_trunc(f0l), vec_trunc(f0h)),
            fp32_to_fp16(vec_trunc(f1l), vec_trunc(f1h))};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE sqrt() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    return {fp32_to_fp16(vec_sqrt(f0l), vec_sqrt(f0h)),
            fp32_to_fp16(vec_sqrt(f1l), vec_sqrt(f1h))};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE reciprocal() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    const vfloat32 one_vec = vec_splats(1.0f);
    return {fp32_to_fp16(vec_div(one_vec, f0l), vec_div(one_vec, f0h)),
            fp32_to_fp16(vec_div(one_vec, f1l), vec_div(one_vec, f1h))};
  }

  Vectorized<c10::Half> C10_ALWAYS_INLINE rsqrt() const {
    vfloat32 f0l, f0h, f1l, f1h;
    fp16_to_fp32(_vec0, f0l, f0h);
    fp16_to_fp32(_vec1, f1l, f1h);
    const vfloat32 one_vec = vec_splats(1.0f);
    return {fp32_to_fp16(vec_div(one_vec, vec_sqrt(f0l)), vec_div(one_vec, vec_sqrt(f0h))),
            fp32_to_fp16(vec_div(one_vec, vec_sqrt(f1l)), vec_div(one_vec, vec_sqrt(f1h)))};
  }

  // frac() is defined after operator- below (requires the free-function
  // template<> operator- to be visible to avoid instantiating the generic one).
  Vectorized<c10::Half> C10_ALWAYS_INLINE frac() const;

  // Sleef-backed math ops: widen to fp32, apply Vectorized<float> method, narrow.
#define VSX_HALF_UNARY(op) \
  Vectorized<c10::Half> op() const { return map_vec_float(&Vectorized<float>::op); }
#define VSX_HALF_BINARY(op, arg) \
  Vectorized<c10::Half> op(const Vectorized<c10::Half>& arg) const { \
    return map2_vec_float(arg, &Vectorized<float>::op); \
  }

  VSX_HALF_UNARY(acos)
  VSX_HALF_UNARY(acosh)
  VSX_HALF_UNARY(asin)
  VSX_HALF_UNARY(asinh)
  VSX_HALF_UNARY(atan)
  VSX_HALF_UNARY(atanh)
  VSX_HALF_BINARY(atan2, b)
  VSX_HALF_BINARY(copysign, sign)
  VSX_HALF_UNARY(erf)
  VSX_HALF_UNARY(erfc)
  VSX_HALF_UNARY(erfinv)
  VSX_HALF_UNARY(exp)
  VSX_HALF_UNARY(exp2)
  VSX_HALF_UNARY(expm1)
  VSX_HALF_UNARY(exp_u20)
  VSX_HALF_UNARY(fexp_u20)
  VSX_HALF_BINARY(fmod, q)
  VSX_HALF_BINARY(hypot, b)
  VSX_HALF_UNARY(i0)
  VSX_HALF_UNARY(i0e)
  VSX_HALF_UNARY(digamma)
  VSX_HALF_BINARY(igamma, x)
  VSX_HALF_BINARY(igammac, x)
  VSX_HALF_UNARY(log)
  VSX_HALF_UNARY(log10)
  VSX_HALF_UNARY(log1p)
  VSX_HALF_UNARY(log2)
  VSX_HALF_BINARY(nextafter, b)
  VSX_HALF_UNARY(sin)
  VSX_HALF_UNARY(sinh)
  VSX_HALF_UNARY(cos)
  VSX_HALF_UNARY(cosh)
  VSX_HALF_UNARY(tan)
  VSX_HALF_UNARY(tanh)
  VSX_HALF_UNARY(lgamma)
  VSX_HALF_BINARY(pow, exp)

#undef VSX_HALF_UNARY
#undef VSX_HALF_BINARY

  // Comparisons must be done in fp32: comparing the raw int16 bits orders
  // negative values wrongly and breaks -0 == +0 and NaN handling.
  // The vbool32 results are packed back into one vbool16 per 8 lanes.
#define VSX_HALF_CMP(op, func)                                              \
  Vectorized<c10::Half> C10_ALWAYS_INLINE op(                               \
      const Vectorized<c10::Half>& other) const {                           \
    vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;                        \
    fp16_to_fp32(_vec0, a0l, a0h);                                          \
    fp16_to_fp32(_vec1, a1l, a1h);                                          \
    fp16_to_fp32(other._vec0, b0l, b0h);                                    \
    fp16_to_fp32(other._vec1, b1l, b1h);                                    \
    return {                                                                \
        vec_pack(func(a0l, b0l), func(a0h, b0h)),                           \
        vec_pack(func(a1l, b1l), func(a1h, b1h))};                          \
  }

  VSX_HALF_CMP(operator==, vec_cmpeq)
  VSX_HALF_CMP(operator!=, vec_cmpne)
  VSX_HALF_CMP(operator<, vec_cmplt)
  VSX_HALF_CMP(operator<=, vec_cmple)
  VSX_HALF_CMP(operator>, vec_cmpgt)
  VSX_HALF_CMP(operator>=, vec_cmpge)
#undef VSX_HALF_CMP
  // eq/ne/lt/le/gt/ge: DEFINE_MEMBER_OP_AND_ONE does not work for c10::Half
  // (vec_splats has no Half overload), so we expand manually.
  Vectorized<c10::Half> C10_ALWAYS_INLINE eq(const Vectorized<c10::Half>& other) const {
    return (*this == other) & Vectorized<c10::Half>(c10::Half(1.0f));
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE ne(const Vectorized<c10::Half>& other) const {
    return (*this != other) & Vectorized<c10::Half>(c10::Half(1.0f));
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE lt(const Vectorized<c10::Half>& other) const {
    return (*this < other) & Vectorized<c10::Half>(c10::Half(1.0f));
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE le(const Vectorized<c10::Half>& other) const {
    return (*this <= other) & Vectorized<c10::Half>(c10::Half(1.0f));
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE gt(const Vectorized<c10::Half>& other) const {
    return (*this > other) & Vectorized<c10::Half>(c10::Half(1.0f));
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE ge(const Vectorized<c10::Half>& other) const {
    return (*this >= other) & Vectorized<c10::Half>(c10::Half(1.0f));
  }

  // Bitwise ops work directly on the int16 representation.
  Vectorized<c10::Half> C10_ALWAYS_INLINE
  operator&(const Vectorized<c10::Half>& other) const {
    return {vec_and(_vec0, other._vec0), vec_and(_vec1, other._vec1)};
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE
  operator|(const Vectorized<c10::Half>& other) const {
    return {vec_or(_vec0, other._vec0), vec_or(_vec1, other._vec1)};
  }
  Vectorized<c10::Half> C10_ALWAYS_INLINE
  operator^(const Vectorized<c10::Half>& other) const {
    return {vec_xor(_vec0, other._vec0), vec_xor(_vec1, other._vec1)};
  }
};

// Free-function arithmetic ops: widen to fp32 inline, compute, narrow.
// GCC VSX built-ins (vec_add, vec_sub, vec_mul, vec_div) cannot be passed as
// function pointers, so each op is inlined directly here.

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator+(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  return {Vectorized<c10::Half>::fp32_to_fp16(vec_add(a0l, b0l), vec_add(a0h, b0h)),
          Vectorized<c10::Half>::fp32_to_fp16(vec_add(a1l, b1l), vec_add(a1h, b1h))};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator-(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  return {Vectorized<c10::Half>::fp32_to_fp16(vec_sub(a0l, b0l), vec_sub(a0h, b0h)),
          Vectorized<c10::Half>::fp32_to_fp16(vec_sub(a1l, b1l), vec_sub(a1h, b1h))};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator*(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  return {Vectorized<c10::Half>::fp32_to_fp16(vec_mul(a0l, b0l), vec_mul(a0h, b0h)),
          Vectorized<c10::Half>::fp32_to_fp16(vec_mul(a1l, b1l), vec_mul(a1h, b1h))};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator/(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  return {Vectorized<c10::Half>::fp32_to_fp16(vec_div(a0l, b0l), vec_div(a0h, b0h)),
          Vectorized<c10::Half>::fp32_to_fp16(vec_div(a1l, b1l), vec_div(a1h, b1h))};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator&(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator|(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE
operator^(const Vectorized<c10::Half>& a, const Vectorized<c10::Half>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

// frac() defined here so that operator- above is already visible, preventing
// the compiler from instantiating the generic operator- from vec_base.h.
Vectorized<c10::Half> C10_ALWAYS_INLINE
Vectorized<c10::Half>::frac() const {
  return *this - trunc();
}

// maximum/minimum: NaN-propagating. vec_max_nan2/vec_min_nan2 are templates
// and cannot be passed as plain function pointers, so inlined directly.
template <>
Vectorized<c10::Half> inline maximum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  return {
      Vectorized<c10::Half>::fp32_to_fp16(vec_max_nan2(a0l, b0l), vec_max_nan2(a0h, b0h)),
      Vectorized<c10::Half>::fp32_to_fp16(vec_max_nan2(a1l, b1l), vec_max_nan2(a1h, b1h))};
}

template <>
Vectorized<c10::Half> inline minimum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  vfloat32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  return {
      Vectorized<c10::Half>::fp32_to_fp16(vec_min_nan2(a0l, b0l), vec_min_nan2(a0h, b0h)),
      Vectorized<c10::Half>::fp32_to_fp16(vec_min_nan2(a1l, b1l), vec_min_nan2(a1h, b1h))};
}

// clamp widens once, applies both bounds in fp32, and narrows once.
template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE clamp(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& min,
    const Vectorized<c10::Half>& max) {
  vfloat32 a0l, a0h, a1l, a1h, n0l, n0h, n1l, n1h, x0l, x0h, x1l, x1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(min.vec0(), n0l, n0h);
  Vectorized<c10::Half>::fp16_to_fp32(min.vec1(), n1l, n1h);
  Vectorized<c10::Half>::fp16_to_fp32(max.vec0(), x0l, x0h);
  Vectorized<c10::Half>::fp16_to_fp32(max.vec1(), x1l, x1h);
  return {
      Vectorized<c10::Half>::fp32_to_fp16(
          vec_min_nan2(x0l, vec_max_nan2(n0l, a0l)),
          vec_min_nan2(x0h, vec_max_nan2(n0h, a0h))),
      Vectorized<c10::Half>::fp32_to_fp16(
          vec_min_nan2(x1l, vec_max_nan2(n1l, a1l)),
          vec_min_nan2(x1h, vec_max_nan2(n1h, a1h)))};
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE clamp_min(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& min) {
  return maximum(min, a);
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE clamp_max(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& max) {
  return minimum(max, a);
}

template <>
Vectorized<c10::Half> C10_ALWAYS_INLINE fmadd(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  vfloat32 a0l, a0h, a1l, a1h;
  vfloat32 b0l, b0h, b1l, b1h;
  vfloat32 c0l, c0h, c1l, c1h;
  Vectorized<c10::Half>::fp16_to_fp32(a.vec0(), a0l, a0h);
  Vectorized<c10::Half>::fp16_to_fp32(a.vec1(), a1l, a1h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec0(), b0l, b0h);
  Vectorized<c10::Half>::fp16_to_fp32(b.vec1(), b1l, b1h);
  Vectorized<c10::Half>::fp16_to_fp32(c.vec0(), c0l, c0h);
  Vectorized<c10::Half>::fp16_to_fp32(c.vec1(), c1l, c1h);
  return {
      Vectorized<c10::Half>::fp32_to_fp16(vec_madd(a0l, b0l, c0l), vec_madd(a0h, b0h, c0h)),
      Vectorized<c10::Half>::fp32_to_fp16(vec_madd(a1l, b1l, c1l), vec_madd(a1h, b1h, c1h))};
}

} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
