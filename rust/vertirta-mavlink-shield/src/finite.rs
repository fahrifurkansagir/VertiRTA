// SPDX-License-Identifier: Apache-2.0
//! Compile-time finite floating-point wrappers eliminating silent NaN/Inf propagation.

/// A strictly finite `f64` value (`!is_nan() && !is_infinite()`).
///
/// Unlike raw IEEE-754 `f64` (where `f64::clamp(NaN, lo, hi)` returns `NaN`),
/// `FiniteF64` enforces finiteness at construction boundaries so downstream flight
/// barriers and geofence math can never silently ingest `NaN` or `±Inf`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct FiniteF64(f64);

impl FiniteF64 {
    pub const ZERO: Self = Self(0.0);

    #[inline]
    pub fn new(v: f64) -> Option<Self> {
        if v.is_finite() {
            Some(Self(v))
        } else {
            None
        }
    }

    #[inline]
    pub fn from_f32(v: f32) -> Option<Self> {
        Self::new(f64::from(v))
    }

    #[inline]
    pub const fn get(self) -> f64 {
        self.0
    }

    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }

    #[inline]
    pub fn clamp_finite(self, lo: f64, hi: f64) -> Self {
        let clamped = self.0.clamp(lo, hi);
        debug_assert!(clamped.is_finite());
        Self(clamped)
    }
}

/// Strictly finite 3D North-East-Down vector in meters or m/s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3Ned {
    pub n: FiniteF64,
    pub e: FiniteF64,
    pub d: FiniteF64,
}

impl Vec3Ned {
    pub const ZERO: Self = Self {
        n: FiniteF64::ZERO,
        e: FiniteF64::ZERO,
        d: FiniteF64::ZERO,
    };

    #[inline]
    pub fn new(n: f64, e: f64, d: f64) -> Option<Self> {
        Some(Self {
            n: FiniteF64::new(n)?,
            e: FiniteF64::new(e)?,
            d: FiniteF64::new(d)?,
        })
    }

    #[inline]
    pub fn from_f32(n: f32, e: f32, d: f32) -> Option<Self> {
        Self::new(f64::from(n), f64::from(e), f64::from(d))
    }

    #[inline]
    pub fn horiz_norm(self) -> f64 {
        self.n.get().hypot(self.e.get())
    }

    #[inline]
    pub fn norm(self) -> f64 {
        let a = self.n.get();
        let b = self.e.get();
        let c = self.d.get();
        (a * a + b * b + c * c).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_f64_rejects_nan_and_infinities() {
        assert!(FiniteF64::new(f64::NAN).is_none());
        assert!(FiniteF64::new(f64::INFINITY).is_none());
        assert!(FiniteF64::new(f64::NEG_INFINITY).is_none());
        assert!(FiniteF64::from_f32(f32::NAN).is_none());
        assert!(Vec3Ned::new(1.0, f64::NAN, -50.0).is_none());

        let v = FiniteF64::new(12.5).expect("12.5 is finite");
        assert_eq!(v.clamp_finite(-10.0, 10.0).get(), 10.0);
    }
}
