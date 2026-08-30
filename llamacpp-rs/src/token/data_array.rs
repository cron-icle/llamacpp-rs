//! an rusty equivalent of `llama_token_data_array`.
use std::ptr;

use crate::{sampling::LlamaSampler, token::data::LlamaTokenData};

use super::LlamaToken;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaTokenDataArray {
    /// the underlying data
    pub data: Vec<LlamaTokenData>,
    /// the index of the selected token in ``data``
    pub selected: Option<usize>,
    /// is the data sorted?
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    /// Create a new `LlamaTokenDataArray` from a vector and whether or not the data is sorted.
    ///
    /// ```
    /// # use llamacpp_rs::token::data::LlamaTokenData;
    /// # use llamacpp_rs::token::data_array::LlamaTokenDataArray;
    /// # use llamacpp_rs::token::LlamaToken;
    /// let array = LlamaTokenDataArray::new(vec![
    ///         LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///         LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    ///    ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    /// ```
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator and whether or not the data is sorted.
    /// ```
    /// # use llamacpp_rs::token::data::LlamaTokenData;
    /// # use llamacpp_rs::token::data_array::LlamaTokenDataArray;
    /// # use llamacpp_rs::token::LlamaToken;
    /// let array = LlamaTokenDataArray::from_iter([
    ///     LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///     LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    /// ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    pub fn from_iter<T>(data: T, sorted: bool) -> LlamaTokenDataArray
    where
        T: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    /// Returns the current selected token, if one exists.
    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }
}

impl LlamaTokenDataArray {
    /// Modify the underlying data as a `llama_token_data_array`. and reconstruct the `LlamaTokenDataArray`.
    ///
    /// # Panics
    ///
    /// Panics if some of the safety conditions are not met. (we cannot check all of them at
    /// runtime so breaking them is UB)
    ///
    /// SAFETY:
    /// The returned array formed by the data pointer and the length must entirely consist of
    /// initialized token data and the length must be less than the capacity of this array's data
    /// buffer.
    /// if the data is not sorted, sorted must be false.
    pub(crate) unsafe fn modify_as_c_llama_token_data_array<T>(
        &mut self,
        modify: impl FnOnce(&mut crate::llama_cpp_sys::llama_token_data_array) -> T,
    ) -> T {
        let size = self.data.len();
        let data = self
            .data
            .as_mut_ptr()
            .cast::<crate::llama_cpp_sys::llama_token_data>();

        let mut c_llama_token_data_array = crate::llama_cpp_sys::llama_token_data_array {
            data,
            size,
            selected: self.selected.and_then(|s| s.try_into().ok()).unwrap_or(-1),
            sorted: self.sorted,
        };

        let result = modify(&mut c_llama_token_data_array);

        assert!(
            c_llama_token_data_array.size <= self.data.capacity(),
            "Size of the returned array exceeds the data buffer's capacity!"
        );
        if !ptr::eq(c_llama_token_data_array.data, data) {
            ptr::copy(
                c_llama_token_data_array.data,
                data,
                c_llama_token_data_array.size,
            );
        }
        self.data.set_len(c_llama_token_data_array.size);

        self.sorted = c_llama_token_data_array.sorted;
        self.selected = c_llama_token_data_array
            .selected
            .try_into()
            .ok()
            .filter(|&s| s < self.data.len());

        result
    }

    /// Modifies the data array by applying a sampler to it
    pub fn apply_sampler(&mut self, sampler: &LlamaSampler) {
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                crate::llama_cpp_sys::llama_sampler_apply(
                    sampler.sampler,
                    c_llama_token_data_array,
                );
            });
        }
    }

    /// Modifies the data array by applying a sampler to it
    #[must_use]
    pub fn with_sampler(mut self, sampler: &mut LlamaSampler) -> Self {
        self.apply_sampler(sampler);
        self
    }

    /// Randomly selects a token from the candidates based on their probabilities.
    ///
    /// # Panics
    /// If the internal llama.cpp sampler fails to select a token.
    pub fn sample_token(&mut self, seed: u32) -> LlamaToken {
        self.apply_sampler(&LlamaSampler::dist(seed));
        self.selected_token()
            .expect("Dist sampler failed to select a token!")
    }

    /// Selects the token with the highest probability.
    ///
    /// # Panics
    /// If the internal llama.cpp sampler fails to select a token.
    pub fn sample_token_greedy(&mut self) -> LlamaToken {
        self.apply_sampler(&LlamaSampler::greedy());
        self.selected_token()
            .expect("Greedy sampler failed to select a token!")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<LlamaTokenData> {
        vec![
            LlamaTokenData::new(LlamaToken::new(0), 0.1, 0.1),
            LlamaTokenData::new(LlamaToken::new(1), 0.9, 0.9),
            LlamaTokenData::new(LlamaToken::new(2), 0.3, 0.3),
        ]
    }

    #[test]
    fn new_stores_data_and_sorted_flag() {
        let array = LlamaTokenDataArray::new(candidates(), false);
        assert_eq!(array.data.len(), 3);
        assert!(!array.sorted);
        assert_eq!(array.selected, None);
    }

    #[test]
    fn from_iter_collects_items() {
        let array = LlamaTokenDataArray::from_iter(candidates(), true);
        assert_eq!(array.data.len(), 3);
        assert!(array.sorted);
    }

    #[test]
    fn selected_token_is_none_before_selection() {
        let array = LlamaTokenDataArray::new(candidates(), false);
        assert_eq!(array.selected_token(), None);
    }

    #[test]
    fn selected_token_reflects_selected_index() {
        let mut array = LlamaTokenDataArray::new(candidates(), false);
        array.selected = Some(1);
        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn sample_token_greedy_picks_highest_logit() {
        let mut array = LlamaTokenDataArray::new(candidates(), false);
        let token = array.sample_token_greedy();
        assert_eq!(token, LlamaToken::new(1));
    }

    #[test]
    fn with_sampler_applies_sampler_and_returns_self() {
        let array = LlamaTokenDataArray::new(candidates(), false);
        let mut sampler = LlamaSampler::greedy();
        let array = array.with_sampler(&mut sampler);
        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn clone_and_debug_and_eq() {
        let array = LlamaTokenDataArray::new(candidates(), false);
        let cloned = array.clone();
        assert_eq!(array, cloned);
        let _ = format!("{array:?}");
    }
}
