//! See [llama-cpp-rs](https://crates.io/crates/llama-cpp-rs) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unpredictable_function_pointer_comparisons)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
