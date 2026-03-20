//! `HuggingFace` tokenizer wrapper.
//!
//! Downloads and caches the tokenizer.json from a `HuggingFace` model
//! repository using hf-hub, then loads it for fast encoding.

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// Load a tokenizer from a `HuggingFace` model repository.
///
/// Downloads `tokenizer.json` on first call; subsequent calls use the cache.
///
/// # Errors
///
/// Returns an error if the tokenizer file cannot be downloaded or parsed.
pub fn load_tokenizer(model_repo: &str) -> crate::Result<Tokenizer> {
    let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
    let repo = api.model(model_repo.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| crate::Error::Download(e.to_string()))?;
    Tokenizer::from_file(tokenizer_path).map_err(|e| crate::Error::Tokenization(e.to_string()))
}

/// Tokenize a query string for embedding, truncating to `model_max_tokens`.
///
/// Returns an [`crate::backend::Encoding`] with `input_ids`, `attention_mask`,
/// and `token_type_ids` cast to `i64`, ready for ONNX inference.
///
/// # Errors
///
/// Returns an error if the tokenizer fails to encode the text.
pub fn tokenize_query(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    model_max_tokens: usize,
) -> crate::Result<crate::backend::Encoding> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| crate::Error::Tokenization(e.to_string()))?;

    let len = encoding.get_ids().len().min(model_max_tokens);
    Ok(crate::backend::Encoding {
        input_ids: encoding.get_ids()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        attention_mask: encoding.get_attention_mask()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
        token_type_ids: encoding.get_type_ids()[..len]
            .iter()
            .map(|&x| i64::from(x))
            .collect(),
    })
}
