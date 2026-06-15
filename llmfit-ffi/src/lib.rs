use llmfit_core::analysis::{InstalledIndex, build_model_fits};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;

// --- State Management ---

pub struct LlmFitContext {
    pub specs: SystemSpecs,
    pub db: ModelDatabase,
    pub installed: InstalledIndex,
}

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|err| {
        *err.borrow_mut() = msg.to_string();
    });
}

fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c_str) => c_str.into_raw(),
        Err(e) => {
            set_error(&format!("Null byte in string: {}", e));
            std::ptr::null_mut()
        }
    }
}

// --- FFI Lifecycle ---

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_create_context() -> *mut LlmFitContext {
    match panic::catch_unwind(|| {
        let specs = SystemSpecs::detect();
        let db = ModelDatabase::embedded();
        let installed = InstalledIndex::empty(); // Future: optionally scan providers here
        Box::into_raw(Box::new(LlmFitContext {
            specs,
            db,
            installed,
        }))
    }) {
        Ok(ptr) => ptr,
        Err(_) => {
            set_error("Panic during context initialization");
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_destroy_context(ctx_ptr: *mut LlmFitContext) {
    if !ctx_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx_ptr);
        }
    }
}

// --- FFI Functions ---

#[unsafe(no_mangle)]
// Clears the stored error after returning it — call at most once per failure.
pub extern "C" fn llmfit_get_last_error() -> *mut c_char {
    let err_msg = LAST_ERROR.with(|err| {
        let mut msg = err.borrow_mut();
        let out = msg.clone();
        msg.clear();
        out
    });

    if err_msg.is_empty() {
        std::ptr::null_mut()
    } else {
        CString::new(err_msg)
            .map(|c| c.into_raw())
            .unwrap_or(std::ptr::null_mut())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_version() -> *mut c_char {
    to_c_string(env!("CARGO_PKG_VERSION").to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_get_system_info(ctx_ptr: *mut LlmFitContext) -> *mut c_char {
    if ctx_ptr.is_null() {
        set_error("Context pointer is null");
        return std::ptr::null_mut();
    }

    match panic::catch_unwind(|| {
        let ctx = unsafe { &*ctx_ptr };
        match serde_json::to_string(&ctx.specs) {
            Ok(json) => to_c_string(json),
            Err(e) => {
                set_error(&format!("JSON serialization failed: {}", e));
                std::ptr::null_mut()
            }
        }
    }) {
        Ok(res) => res,
        Err(_) => {
            set_error("Panic during system info retrieval");
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_recommend_models(ctx_ptr: *mut LlmFitContext, limit: u32) -> *mut c_char {
    if ctx_ptr.is_null() {
        set_error("Context pointer is null");
        return std::ptr::null_mut();
    }

    match panic::catch_unwind(|| {
        let ctx = unsafe { &*ctx_ptr };
        let mut fits = build_model_fits(&ctx.db, &ctx.specs, &ctx.installed, None, None);

        fits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // limit = 0 means no cap — return all results.
        let limit_idx = limit as usize;
        if limit_idx > 0 && fits.len() > limit_idx {
            fits.truncate(limit_idx);
        }

        match serde_json::to_string(&fits) {
            Ok(json) => to_c_string(json),
            Err(e) => {
                set_error(&format!("JSON serialization failed: {}", e));
                std::ptr::null_mut()
            }
        }
    }) {
        Ok(res) => res,
        Err(_) => {
            set_error("Panic during model recommendation");
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn llmfit_get_model_info(
    ctx_ptr: *mut LlmFitContext,
    model_name: *const c_char,
) -> *mut c_char {
    if ctx_ptr.is_null() {
        set_error("Context pointer is null");
        return std::ptr::null_mut();
    }
    if model_name.is_null() {
        set_error("Model name pointer is null");
        return std::ptr::null_mut();
    }

    match panic::catch_unwind(|| {
        let ctx = unsafe { &*ctx_ptr };
        let c_str = unsafe { CStr::from_ptr(model_name) };
        let name = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_error(&format!("Invalid UTF-8 in model name: {}", e));
                return std::ptr::null_mut();
            }
        };

        let name_lower = name.to_lowercase();
        let models: Vec<_> = ctx
            .db
            .get_all_models()
            .into_iter()
            .filter(|m| m.name.to_lowercase().contains(&name_lower))
            .collect();

        let mut fits = Vec::new();
        for m in models {
            if llmfit_core::fit::backend_compatible(&m, &ctx.specs) {
                let mut fit = llmfit_core::fit::ModelFit::analyze(&m, &ctx.specs);
                fit.installed = ctx.installed.is_installed(&m.name);
                fits.push(fit);
            }
        }

        match serde_json::to_string(&fits) {
            Ok(json) => to_c_string(json),
            Err(e) => {
                set_error(&format!("JSON serialization failed: {}", e));
                std::ptr::null_mut()
            }
        }
    }) {
        Ok(res) => res,
        Err(_) => {
            set_error("Panic during model info retrieval");
            std::ptr::null_mut()
        }
    }
}
