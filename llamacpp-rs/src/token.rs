//! Safe wrappers around `llama_token_data` and `llama_token_data_array`.

use std::fmt::Debug;
use std::fmt::Display;

pub mod data;
pub mod data_array;
pub mod logit_bias;

/// A safe wrapper for `llama_token`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaToken(pub llama_cpp_sys::llama_token);

impl Display for LlamaToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LlamaToken {
    /// Create a new `LlamaToken` from a i32.
    ///
    /// ```
    /// # use llamacpp_rs::token::LlamaToken;
    /// let token = LlamaToken::new(0);
    /// assert_eq!(token, LlamaToken(0));
    /// ```
    #[must_use]
    pub fn new(token_id: i32) -> Self {
        Self(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_wraps_the_id() {
        assert_eq!(LlamaToken::new(42), LlamaToken(42));
    }

    #[test]
    fn display_prints_the_inner_id() {
        assert_eq!(format!("{}", LlamaToken::new(7)), "7");
    }

    #[test]
    fn ordering_and_equality() {
        assert!(LlamaToken::new(1) < LlamaToken::new(2));
        assert_eq!(LlamaToken::new(3), LlamaToken::new(3));
        assert_ne!(LlamaToken::new(3), LlamaToken::new(4));
    }

    #[test]
    fn debug_and_hash_are_derived() {
        let token = LlamaToken::new(1);
        let _ = format!("{token:?}");
        let copied = token;
        assert_eq!(token, copied);
    }
}
