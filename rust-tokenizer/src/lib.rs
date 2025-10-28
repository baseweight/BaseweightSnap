use std::ffi::{CStr, CString};
use std::os::raw::{c_char};
use std::ptr;
use tokenizers::Tokenizer;

/// Opaque handle to a tokenizer
pub struct TokenizerHandle {
    tokenizer: Tokenizer,
    image_token_id: u32,
    image_token: String,
}

/// C-compatible struct for returning tokenization results
#[repr(C)]
pub struct TokenizationResult {
    pub token_ids: *mut i64,
    pub num_tokens: usize,
    pub image_token_positions: *mut usize,
    pub num_image_tokens: usize,
}

/// Load a tokenizer from a JSON file and return an opaque handle
///
/// # Safety
/// - `tokenizer_path` must be a valid null-terminated C string
/// - `image_token` must be a valid null-terminated C string
/// - Returns null pointer on failure
#[no_mangle]
pub unsafe extern "C" fn nanovlm_load_tokenizer(
    tokenizer_path: *const c_char,
    image_token: *const c_char,
) -> *mut TokenizerHandle {
    if tokenizer_path.is_null() || image_token.is_null() {
        return ptr::null_mut();
    }

    let path_cstr = unsafe { CStr::from_ptr(tokenizer_path) };
    let path = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let image_token_cstr = unsafe { CStr::from_ptr(image_token) };
    let img_token = match image_token_cstr.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut(),
    };

    let mut tokenizer = match Tokenizer::from_file(path) {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };

    // Add the image token as a special token (like Python's extra_special_tokens)
    use tokenizers::AddedToken;

    // Add tokens in same order as Python to get consistent token IDs
    // Python's vlm_extra_tokens dict has keys: image_token, global_image_token, r1c1, r1c2, ...
    let mut special_tokens = vec![
        // First: main image token
        AddedToken::from(img_token.clone(), true),
        // Second: global image token
        AddedToken::from("<|global_image|>".to_string(), true),
    ];

    // Third: all row/col tokens (8x8 grid max)
    for row in 1..=8 {
        for col in 1..=8 {
            let token = format!("<row_{}_col_{}>", row, col);
            special_tokens.push(AddedToken::from(token, true));
        }
    }

    tokenizer.add_special_tokens(&special_tokens);

    // Now get the ID of the image token
    let image_token_id = match tokenizer.token_to_id(&img_token) {
        Some(id) => id,
        None => return ptr::null_mut(),
    };

    let handle = Box::new(TokenizerHandle {
        tokenizer,
        image_token_id,
        image_token: img_token,
    });

    Box::into_raw(handle)
}

/// Free a tokenizer handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `nanovlm_load_tokenizer`
/// - After calling this, the handle must not be used again
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_tokenizer(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        let _ = unsafe { Box::from_raw(handle) };
    }
}

/// Tokenize text with image token placeholders
///
/// # Safety
/// - `handle` must be a valid tokenizer handle
/// - `text` must be a valid null-terminated C string
/// - `image_token_length` specifies how many times to repeat the image token
/// - Caller must free the result using `nanovlm_free_tokenization_result`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_tokenize(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    image_token_length: usize,
) -> TokenizationResult {
    if handle.is_null() || text.is_null() {
        return TokenizationResult {
            token_ids: ptr::null_mut(),
            num_tokens: 0,
            image_token_positions: ptr::null_mut(),
            num_image_tokens: 0,
        };
    }

    let tokenizer_handle = unsafe { &*handle };
    let text_cstr = unsafe { CStr::from_ptr(text) };
    let prompt = match text_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return TokenizationResult {
            token_ids: ptr::null_mut(),
            num_tokens: 0,
            image_token_positions: ptr::null_mut(),
            num_image_tokens: 0,
        },
    };

    // Create full prompt with image tokens
    let image_tokens_repeated = tokenizer_handle.image_token.repeat(image_token_length);
    let full_prompt = format!("{}{}", image_tokens_repeated, prompt);

    // Encode
    let encoding = match tokenizer_handle.tokenizer.encode(full_prompt, false) {
        Ok(enc) => enc,
        Err(_) => return TokenizationResult {
            token_ids: ptr::null_mut(),
            num_tokens: 0,
            image_token_positions: ptr::null_mut(),
            num_image_tokens: 0,
        },
    };

    let token_ids = encoding.get_ids();

    // Find positions for ONLY <|image|> tokens
    // Special tokens like <|global_image|> and <row_X_col_Y> are context tokens
    // that should NOT be replaced with embeddings - they tell the model which patch to expect
    let mut image_positions = Vec::new();
    for (idx, &token_id) in token_ids.iter().enumerate() {
        if token_id == tokenizer_handle.image_token_id {
            image_positions.push(idx);
        }
    }

    // Convert to i64 for compatibility with ExecuTorch
    let token_ids_i64: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
    let num_tokens = token_ids_i64.len();
    let num_image_tokens = image_positions.len();

    // Allocate and copy token IDs
    let mut token_ids_boxed = token_ids_i64.into_boxed_slice();
    let token_ids_ptr = token_ids_boxed.as_mut_ptr();
    std::mem::forget(token_ids_boxed);

    // Allocate and copy image positions
    let mut image_positions_boxed = image_positions.into_boxed_slice();
    let image_positions_ptr = image_positions_boxed.as_mut_ptr();
    std::mem::forget(image_positions_boxed);

    TokenizationResult {
        token_ids: token_ids_ptr,
        num_tokens,
        image_token_positions: image_positions_ptr,
        num_image_tokens,
    }
}

/// Free tokenization result
///
/// # Safety
/// - `result` must be a valid TokenizationResult returned by `nanovlm_tokenize`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_tokenization_result(result: TokenizationResult) {
    if !result.token_ids.is_null() {
        let _ = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(result.token_ids, result.num_tokens)) };
    }
    if !result.image_token_positions.is_null() {
        let _ = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(result.image_token_positions, result.num_image_tokens)) };
    }
}

/// Decode token IDs back to text
///
/// # Safety
/// - `handle` must be a valid tokenizer handle
/// - `token_ids` must be a valid array of `num_tokens` elements
/// - Returns a newly allocated C string that must be freed with `nanovlm_free_string`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_decode(
    handle: *mut TokenizerHandle,
    token_ids: *const i64,
    num_tokens: usize,
) -> *mut c_char {
    if handle.is_null() || token_ids.is_null() {
        return ptr::null_mut();
    }

    let tokenizer_handle = unsafe { &*handle };
    let ids_slice = unsafe { std::slice::from_raw_parts(token_ids, num_tokens) };
    let ids_u32: Vec<u32> = ids_slice.iter().map(|&id| id as u32).collect();

    let text = match tokenizer_handle.tokenizer.decode(&ids_u32, true) {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };

    match CString::new(text) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by nanovlm_decode
///
/// # Safety
/// - `str` must be a valid pointer returned by `nanovlm_decode`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_string(str: *mut c_char) {
    if !str.is_null() {
        let _ = unsafe { CString::from_raw(str) };
    }
}