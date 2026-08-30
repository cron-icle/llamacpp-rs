//! Representation of an initialized llama backend

use crate::llama_cpp_sys::ggml_log_level;
use crate::LlamaCppError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

/// Representation of an initialized llama backend
/// This is required as a parameter for most llama functions as the backend must be initialized
/// before any llama functions are called. This type is proof of initialization.
#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    /// Mark the llama backend as initialized
    fn mark_init() -> crate::Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => Err(LlamaCppError::BackendAlreadyInitialized),
        }
    }

    /// Initialize the llama backend (without numa).
    ///
    /// # Examples
    ///
    /// ```
    ///# use llamacpp_rs::llama_backend::LlamaBackend;
    ///# use llamacpp_rs::LlamaCppError;
    ///# use std::error::Error;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    ///
    /// let backend = LlamaBackend::init()?;
    /// // the llama backend can only be initialized once
    /// assert_eq!(Err(LlamaCppError::BackendAlreadyInitialized), LlamaBackend::init());
    ///
    ///# Ok(())
    ///# }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn init() -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe { crate::llama_cpp_sys::llama_backend_init() }
        Ok(LlamaBackend {})
    }

    /// Initialize the llama backend (with numa).
    /// ```
    ///# use llamacpp_rs::llama_backend::LlamaBackend;
    ///# use std::error::Error;
    ///# use llamacpp_rs::llama_backend::NumaStrategy;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    /// let llama_backend = LlamaBackend::init_numa(NumaStrategy::MIRROR)?;
    ///
    ///# Ok(())
    ///# }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn init_numa(strategy: NumaStrategy) -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe {
            crate::llama_cpp_sys::llama_numa_init(crate::llama_cpp_sys::ggml_numa_strategy::from(
                strategy,
            ));
        }
        Ok(LlamaBackend {})
    }

    /// Was the code built for a GPU backend & is a supported one available.
    pub fn supports_gpu_offload(&self) -> bool {
        unsafe { crate::llama_cpp_sys::llama_supports_gpu_offload() }
    }

    /// Does this platform support loading the model via mmap.
    pub fn supports_mmap(&self) -> bool {
        unsafe { crate::llama_cpp_sys::llama_supports_mmap() }
    }

    /// Does this platform support locking the model in RAM.
    pub fn supports_mlock(&self) -> bool {
        unsafe { crate::llama_cpp_sys::llama_supports_mlock() }
    }

    /// Change the output of llama.cpp's logging to be voided instead of pushed to `stderr`.
    pub fn void_logs(&mut self) {
        unsafe extern "C" fn void_log(
            _level: ggml_log_level,
            _text: *const ::std::os::raw::c_char,
            _user_data: *mut ::std::os::raw::c_void,
        ) {
        }

        unsafe {
            crate::llama_cpp_sys::llama_log_set(Some(void_log), std::ptr::null_mut());
        }
    }
}

/// A rusty wrapper around `numa_strategy`.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum NumaStrategy {
    /// The numa strategy is disabled.
    DISABLED,
    /// help wanted: what does this do?
    DISTRIBUTE,
    /// help wanted: what does this do?
    ISOLATE,
    /// help wanted: what does this do?
    NUMACTL,
    /// help wanted: what does this do?
    MIRROR,
    /// help wanted: what does this do?
    COUNT,
}

/// An invalid numa strategy was provided.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct InvalidNumaStrategy(
    /// The invalid numa strategy that was provided.
    pub crate::llama_cpp_sys::ggml_numa_strategy,
);

impl TryFrom<crate::llama_cpp_sys::ggml_numa_strategy> for NumaStrategy {
    type Error = InvalidNumaStrategy;

    fn try_from(value: crate::llama_cpp_sys::ggml_numa_strategy) -> Result<Self, Self::Error> {
        match value {
            crate::llama_cpp_sys::GGML_NUMA_STRATEGY_DISABLED => Ok(Self::DISABLED),
            crate::llama_cpp_sys::GGML_NUMA_STRATEGY_DISTRIBUTE => Ok(Self::DISTRIBUTE),
            crate::llama_cpp_sys::GGML_NUMA_STRATEGY_ISOLATE => Ok(Self::ISOLATE),
            crate::llama_cpp_sys::GGML_NUMA_STRATEGY_NUMACTL => Ok(Self::NUMACTL),
            crate::llama_cpp_sys::GGML_NUMA_STRATEGY_MIRROR => Ok(Self::MIRROR),
            crate::llama_cpp_sys::GGML_NUMA_STRATEGY_COUNT => Ok(Self::COUNT),
            value => Err(InvalidNumaStrategy(value)),
        }
    }
}

impl From<NumaStrategy> for crate::llama_cpp_sys::ggml_numa_strategy {
    fn from(value: NumaStrategy) -> Self {
        match value {
            NumaStrategy::DISABLED => crate::llama_cpp_sys::GGML_NUMA_STRATEGY_DISABLED,
            NumaStrategy::DISTRIBUTE => crate::llama_cpp_sys::GGML_NUMA_STRATEGY_DISTRIBUTE,
            NumaStrategy::ISOLATE => crate::llama_cpp_sys::GGML_NUMA_STRATEGY_ISOLATE,
            NumaStrategy::NUMACTL => crate::llama_cpp_sys::GGML_NUMA_STRATEGY_NUMACTL,
            NumaStrategy::MIRROR => crate::llama_cpp_sys::GGML_NUMA_STRATEGY_MIRROR,
            NumaStrategy::COUNT => crate::llama_cpp_sys::GGML_NUMA_STRATEGY_COUNT,
        }
    }
}

/// Drops the llama backend.
/// ```
///
///# use llamacpp_rs::llama_backend::LlamaBackend;
///# use std::error::Error;
///
///# fn main() -> Result<(), Box<dyn Error>> {
/// let backend = LlamaBackend::init()?;
/// drop(backend);
/// // can be initialized again after being dropped
/// let backend = LlamaBackend::init()?;
///# Ok(())
///# }
///
/// ```
impl Drop for LlamaBackend {
    fn drop(&mut self) {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(true, false, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(_) => {
                unreachable!("This should not be reachable as the only ways to obtain a llama backend involve marking the backend as initialized.")
            }
        }
        unsafe { crate::llama_cpp_sys::llama_backend_free() }
    }
}

/// Compile-time path to the built GGML backend modules directory.
/// Populated by build.rs from `DEP_LLAMA_BACKENDS_DIR` (emitted by llama-cpp-rs-sys).
/// None on static builds or when the feature is disabled.
#[cfg(feature = "dynamic-backends")]
pub const BACKENDS_DIR: Option<&str> = option_env!("GGML_BACKENDS_DIR");

/// Load GGML backend modules from the given directory.
///
/// Call this before [`LlamaBackend::init`] to enable runtime hardware selection
/// (Vulkan, CPU-AVX512, CPU-AVX2, etc.) when built with the `dynamic-backends` feature.
#[cfg(feature = "dynamic-backends")]
pub fn load_backends_from_path(path: &std::path::Path) {
    let s = std::ffi::CString::new(path.to_str().expect("path must be valid UTF-8"))
        .expect("path must not contain null bytes");
    unsafe { crate::llama_cpp_sys::ggml_backend_load_all_from_path(s.as_ptr()) }
}

/// Load GGML backend modules from the compile-time default directory ([`BACKENDS_DIR`]).
///
/// This is a no-op when `BACKENDS_DIR` is `None` (static builds or development builds
/// that have not set `GGML_BACKENDS_DIR`).
#[cfg(feature = "dynamic-backends")]
pub fn load_backends() {
    if let Some(dir) = BACKENDS_DIR {
        load_backends_from_path(std::path::Path::new(dir));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_from_and_to() {
        let numas = [
            NumaStrategy::DISABLED,
            NumaStrategy::DISTRIBUTE,
            NumaStrategy::ISOLATE,
            NumaStrategy::NUMACTL,
            NumaStrategy::MIRROR,
            NumaStrategy::COUNT,
        ];

        for numa in &numas {
            let from = crate::llama_cpp_sys::ggml_numa_strategy::from(*numa);
            let to = NumaStrategy::try_from(from).expect("Failed to convert from and to");
            assert_eq!(*numa, to);
        }
    }

    #[test]
    fn check_invalid_numa() {
        let invalid = 800;
        let invalid = NumaStrategy::try_from(invalid);
        assert_eq!(invalid, Err(InvalidNumaStrategy(invalid.unwrap_err().0)));
    }

    #[test]
    fn init_reports_capabilities_and_can_void_logs() {
        let mut backend = match LlamaBackend::init() {
            Ok(backend) => backend,
            // Another test in this binary (or a doctest) may have already
            // initialized the backend; the capability queries below don't
            // require ownership of a fresh backend to be meaningful.
            Err(crate::LlamaCppError::BackendAlreadyInitialized) => return,
            Err(e) => panic!("unexpected backend init error: {e:?}"),
        };

        // These are just smoke checks: the values are platform-dependent, so
        // we only assert the calls complete without panicking/crashing.
        let _ = backend.supports_gpu_offload();
        let _ = backend.supports_mmap();
        let _ = backend.supports_mlock();
        backend.void_logs();

        drop(backend);

        // Backend can be re-initialized after being dropped.
        let backend = LlamaBackend::init().expect("backend should be reinitializable");
        drop(backend);
    }
}
