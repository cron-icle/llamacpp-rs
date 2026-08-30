//! Utilities for working with `llama_token_type` values.
use enumflags2::{bitflags, BitFlags};
use std::ops::{Deref, DerefMut};

/// A rust flavored equivalent of `llama_token_type`.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
#[allow(clippy::module_name_repetitions, missing_docs)]
pub enum LlamaTokenAttr {
    Unknown = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_UNKNOWN as _,
    Unused = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_UNUSED as _,
    Normal = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_NORMAL as _,
    Control = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_CONTROL as _,
    UserDefined = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_USER_DEFINED as _,
    Byte = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_BYTE as _,
    Normalized = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_NORMALIZED as _,
    LStrip = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_LSTRIP as _,
    RStrip = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_RSTRIP as _,
    SingleWord = crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_SINGLE_WORD as _,
}

/// A set of `LlamaTokenAttrs`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaTokenAttrs(pub BitFlags<LlamaTokenAttr>);

impl Deref for LlamaTokenAttrs {
    type Target = BitFlags<LlamaTokenAttr>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LlamaTokenAttrs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TryFrom<crate::llama_cpp_sys::llama_token_type> for LlamaTokenAttrs {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: crate::llama_cpp_sys::llama_vocab_type) -> Result<Self, Self::Error> {
        Ok(Self(BitFlags::from_bits(value as _).map_err(|e| {
            LlamaTokenTypeFromIntError::UnknownValue(e.invalid_bits())
        })?))
    }
}

/// An error type for `LlamaTokenType::try_from`.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenTypeFromIntError {
    /// The value is not a valid `llama_token_type`.
    #[error("Unknown Value {0}")]
    UnknownValue(std::ffi::c_uint),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_bitflag_values_round_trip() {
        let attrs = LlamaTokenAttrs::try_from(crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_NORMAL)
            .expect("valid attr value should convert");
        assert!(attrs.contains(LlamaTokenAttr::Normal));
    }

    #[test]
    fn invalid_bitflag_value_errors() {
        let err = LlamaTokenAttrs::try_from(1 << 31).unwrap_err();
        assert_eq!(err, LlamaTokenTypeFromIntError::UnknownValue(1 << 31));
        assert_eq!(format!("{err}"), format!("Unknown Value {}", 1u32 << 31));
    }

    #[test]
    fn deref_and_deref_mut_expose_bitflags() {
        let mut attrs = LlamaTokenAttrs::try_from(crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_NORMAL)
            .expect("valid attr value should convert");
        assert!(attrs.contains(LlamaTokenAttr::Normal));
        attrs.insert(LlamaTokenAttr::Control);
        assert!(attrs.contains(LlamaTokenAttr::Control));
    }

    #[test]
    fn clone_copy_debug_eq() {
        let attrs = LlamaTokenAttrs::try_from(crate::llama_cpp_sys::LLAMA_TOKEN_ATTR_NORMAL)
            .expect("valid attr value should convert");
        let copied = attrs;
        let cloned = attrs.clone();
        assert_eq!(attrs, copied);
        assert_eq!(attrs, cloned);
        let _ = format!("{attrs:?}");
    }
}
