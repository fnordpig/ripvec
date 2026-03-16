//! HuggingFace tokenizer wrapper.
//!
//! Downloads and caches the tokenizer.json from a HuggingFace model
//! repository using hf-hub, then loads it for fast encoding.

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// Load a tokenizer from a HuggingFace model repository.
///
/// Downloads `tokenizer.json` on first call; subsequent calls use the cache.
pub fn load_tokenizer(model_repo: &str) -> crate::Result<Tokenizer> {
    let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
    let repo = api.model(model_repo.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| crate::Error::Download(e.to_string()))?;
    Tokenizer::from_file(tokenizer_path).map_err(|e| crate::Error::Tokenization(e.to_string()))
}
