//! Platform abstraction for wall-clock deadlines.
//!
//! * native std: `std::time::Instant`
//! * wasm32 + std: `web_time::Instant` (std's Instant panics in browsers)
//! * no_std: a zero-sized stub; deadlines never expire and the solver stops
//!   on its convergence criteria and finite kick budgets instead.

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
pub use std::time::Instant;

#[cfg(all(feature = "std", target_arch = "wasm32"))]
pub use web_time::Instant;

#[cfg(not(feature = "std"))]
pub use stub::Instant;

/// Has the deadline passed? Always false without an OS clock.
#[inline]
pub fn expired(deadline: Instant) -> bool {
    #[cfg(feature = "std")]
    {
        Instant::now() >= deadline
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = deadline;
        false
    }
}

/// Earlier of two deadlines.
#[inline]
pub fn earlier(a: Instant, b: Instant) -> Instant {
    #[cfg(feature = "std")]
    {
        a.min(b)
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = b;
        a
    }
}

#[cfg(not(feature = "std"))]
mod stub {
    use core::ops::Add;
    use core::time::Duration;

    /// Placeholder clock for no_std builds. All instants are equal; deadline
    /// checks go through [`super::expired`], which always returns false.
    #[derive(Clone, Copy, Debug)]
    pub struct Instant;

    impl Instant {
        pub fn now() -> Self {
            Instant
        }

        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }

        pub fn saturating_duration_since(&self, _earlier: Instant) -> Duration {
            Duration::ZERO
        }
    }

    impl Add<Duration> for Instant {
        type Output = Instant;
        fn add(self, _rhs: Duration) -> Instant {
            Instant
        }
    }
}
