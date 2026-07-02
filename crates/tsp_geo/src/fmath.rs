//! Float math shim: std intrinsics when available, libm otherwise.

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!("tsp-geo requires either the `std` feature or the `libm` feature");

#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

#[inline(always)]
pub fn sin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sin(x)
    }
}

#[inline(always)]
pub fn cos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cos(x)
    }
}

#[inline(always)]
pub fn sin_cos(x: f64) -> (f64, f64) {
    #[cfg(feature = "std")]
    {
        x.sin_cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sincos(x)
    }
}

#[inline(always)]
pub fn asin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.asin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::asin(x)
    }
}
