//! Key-value overrides for a model.

use crate::model::params::LlamaModelParams;
use std::ffi::{CStr, CString};
use std::fmt::Debug;

/// An override value for a model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamOverrideValue {
    /// A string value
    Bool(bool),
    /// A float value
    Float(f64),
    /// A integer value
    Int(i64),
    /// A string value
    Str([std::os::raw::c_char; 128]),
}

impl ParamOverrideValue {
    pub(crate) fn tag(&self) -> llama_cpp_sys::llama_model_kv_override_type {
        match self {
            ParamOverrideValue::Bool(_) => llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            ParamOverrideValue::Float(_) => llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            ParamOverrideValue::Int(_) => llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            ParamOverrideValue::Str(_) => llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
        }
    }

    pub(crate) fn value(&self) -> llama_cpp_sys::llama_model_kv_override__bindgen_ty_1 {
        match self {
            ParamOverrideValue::Bool(value) => {
                llama_cpp_sys::llama_model_kv_override__bindgen_ty_1 { val_bool: *value }
            }
            ParamOverrideValue::Float(value) => {
                llama_cpp_sys::llama_model_kv_override__bindgen_ty_1 { val_f64: *value }
            }
            ParamOverrideValue::Int(value) => {
                llama_cpp_sys::llama_model_kv_override__bindgen_ty_1 { val_i64: *value }
            }
            ParamOverrideValue::Str(c_string) => {
                llama_cpp_sys::llama_model_kv_override__bindgen_ty_1 { val_str: *c_string }
            }
        }
    }
}

impl From<&llama_cpp_sys::llama_model_kv_override> for ParamOverrideValue {
    fn from(
        llama_cpp_sys::llama_model_kv_override {
            key: _,
            tag,
            __bindgen_anon_1,
        }: &llama_cpp_sys::llama_model_kv_override,
    ) -> Self {
        match *tag {
            llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_INT => {
                ParamOverrideValue::Int(unsafe { __bindgen_anon_1.val_i64 })
            }
            llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT => {
                ParamOverrideValue::Float(unsafe { __bindgen_anon_1.val_f64 })
            }
            llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL => {
                ParamOverrideValue::Bool(unsafe { __bindgen_anon_1.val_bool })
            }
            llama_cpp_sys::LLAMA_KV_OVERRIDE_TYPE_STR => {
                ParamOverrideValue::Str(unsafe { __bindgen_anon_1.val_str })
            }
            _ => unreachable!("Unknown tag of {tag}"),
        }
    }
}

/// A struct implementing [`IntoIterator`] over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrides<'a> {
    model_params: &'a LlamaModelParams,
}

impl KvOverrides<'_> {
    pub(super) fn new<'a>(model_params: &'a LlamaModelParams) -> KvOverrides<'a> {
        KvOverrides { model_params }
    }
}

impl<'a> IntoIterator for KvOverrides<'a> {
    // I'm fairly certain this could be written returning by reference, but I'm not sure how to do it safely. I do not
    // expect this to be a performance bottleneck so the copy should be fine. (let me know if it's not fine!)
    type Item = (CString, ParamOverrideValue);
    type IntoIter = KvOverrideValueIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        KvOverrideValueIterator {
            model_params: self.model_params,
            current: 0,
        }
    }
}

/// An iterator over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrideValueIterator<'a> {
    model_params: &'a LlamaModelParams,
    current: usize,
}

impl Iterator for KvOverrideValueIterator<'_> {
    type Item = (CString, ParamOverrideValue);

    fn next(&mut self) -> Option<Self::Item> {
        let overrides = self.model_params.params.kv_overrides;
        if overrides.is_null() {
            return None;
        }

        // SAFETY: llama.cpp seems to guarantee that the last element contains an empty key or is valid. We've checked
        // the prev one in the last iteration, the next one should be valid or 0 (and thus safe to deref)
        let current = unsafe { *overrides.add(self.current) };

        if current.key[0] == 0 {
            return None;
        }

        let value = ParamOverrideValue::from(&current);

        let key = unsafe { CStr::from_ptr(current.key.as_ptr()).to_owned() };

        self.current += 1;
        Some((key, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::params::LlamaModelParams;
    use std::pin::pin;

    #[test]
    fn empty_overrides_iterate_to_nothing() {
        let params = Box::pin(LlamaModelParams::default());
        assert_eq!(params.kv_overrides().into_iter().count(), 0);
    }

    #[test]
    fn bool_override_round_trips() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("flag").unwrap();
        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Bool(true));

        let overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].0, key);
        assert_eq!(overrides[0].1, ParamOverrideValue::Bool(true));
    }

    #[test]
    fn float_override_round_trips() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("ratio").unwrap();
        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Float(2.5));

        let overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
        assert_eq!(overrides[0].1, ParamOverrideValue::Float(2.5));
    }

    #[test]
    fn str_override_round_trips() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("name").unwrap();
        let mut buf = [0 as std::os::raw::c_char; 128];
        for (i, b) in b"hello".iter().enumerate() {
            buf[i] = *b as std::os::raw::c_char;
        }
        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Str(buf));

        let overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
        assert_eq!(overrides[0].1, ParamOverrideValue::Str(buf));
    }

    #[test]
    fn multiple_overrides_all_present() {
        let mut params = pin!(LlamaModelParams::default());
        params
            .as_mut()
            .append_kv_override(&CString::new("a").unwrap(), ParamOverrideValue::Int(1));
        params
            .as_mut()
            .append_kv_override(&CString::new("b").unwrap(), ParamOverrideValue::Int(2));

        let overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
        assert_eq!(overrides.len(), 2);
        assert_eq!(overrides[0].1, ParamOverrideValue::Int(1));
        assert_eq!(overrides[1].1, ParamOverrideValue::Int(2));
    }

    #[test]
    fn debug_impls_do_not_panic() {
        let params = Box::pin(LlamaModelParams::default());
        let kv_overrides = params.kv_overrides();
        let _ = format!("{kv_overrides:?}");
        let iter = kv_overrides.into_iter();
        let _ = format!("{iter:?}");
    }
}
