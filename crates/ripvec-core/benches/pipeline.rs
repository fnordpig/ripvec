//! Benchmarks for the ripvec embedding pipeline.
//!
//! Covers: tokenization, embedding (single + batch), chunking,
//! similarity, and the full search pipeline.

#![expect(
    clippy::format_collect,
    clippy::cast_precision_loss,
    reason = "benchmark helper code — clarity over micro-optimization"
)]

use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ripvec_core::backend::{BackendKind, DeviceHint, EmbedBackend, Encoding};
use ripvec_core::chunk::ChunkConfig;
use ripvec_core::embed::{SearchConfig, DEFAULT_BATCH_SIZE};
use ripvec_core::profile::Profiler;

const MODEL_REPO: &str = "BAAI/bge-small-en-v1.5";

/// Shared test fixtures loaded once per benchmark group.
struct Fixtures {
    backend: Box<dyn EmbedBackend>,
    tokenizer: tokenizers::Tokenizer,
    /// A batch of pre-tokenized encodings (short sequences ~64 tokens).
    short_encodings: Vec<Encoding>,
    /// A batch of pre-tokenized encodings (long sequences ~256 tokens).
    long_encodings: Vec<Encoding>,
    /// Sample Rust source for chunking benchmarks.
    rust_source: String,
}

impl Fixtures {
    fn load() -> Self {
        let backend = ripvec_core::backend::load_backend(
            BackendKind::Cpu,
            MODEL_REPO,
            DeviceHint::Cpu,
            &ripvec_core::backend::InferenceOpts::default(),
        )
        .expect("backend load");
        let tokenizer = ripvec_core::tokenize::load_tokenizer(MODEL_REPO).expect("tokenizer");

        // Generate short and long text samples
        let short_text = "fn hello_world() { println!(\"Hello, world!\"); }";
        let long_text = (0..20)
            .map(|i| format!("fn function_{i}(x: i32, y: i32) -> i32 {{ x + y + {i} }}\n"))
            .collect::<String>();

        let tokenize = |text: &str, max_len: usize| -> Encoding {
            let enc = tokenizer.encode(text, true).expect("tokenize");
            let ids = enc.get_ids();
            let len = ids.len().min(max_len);
            Encoding {
                input_ids: ids[..len].iter().map(|&id| i64::from(id)).collect(),
                attention_mask: vec![1i64; len],
                token_type_ids: vec![0i64; len],
            }
        };

        let short_enc = tokenize(short_text, 512);
        let long_enc = tokenize(&long_text, 256);

        // Build batches of 32
        let short_encodings: Vec<Encoding> = (0..DEFAULT_BATCH_SIZE)
            .map(|_| Encoding {
                input_ids: short_enc.input_ids.clone(),
                attention_mask: short_enc.attention_mask.clone(),
                token_type_ids: short_enc.token_type_ids.clone(),
            })
            .collect();

        let long_encodings: Vec<Encoding> = (0..DEFAULT_BATCH_SIZE)
            .map(|_| Encoding {
                input_ids: long_enc.input_ids.clone(),
                attention_mask: long_enc.attention_mask.clone(),
                token_type_ids: long_enc.token_type_ids.clone(),
            })
            .collect();

        // Sample Rust source for chunking
        let rust_source = (0..50)
            .map(|i| {
                format!(
                    "/// Doc comment for function {i}.\n\
                     pub fn compute_{i}(input: &[f32]) -> Vec<f32> {{\n\
                         input.iter().map(|x| x * {i}.0 + 1.0).collect()\n\
                     }}\n\n"
                )
            })
            .collect::<String>();

        Self {
            backend,
            tokenizer,
            short_encodings,
            long_encodings,
            rust_source,
        }
    }
}

// --- Embedding benchmarks ---

fn bench_embed_batch(c: &mut Criterion) {
    let f = Fixtures::load();
    let mut group = c.benchmark_group("embed_batch");

    for (name, encodings) in [
        ("short_x1", &f.short_encodings[..1]),
        ("short_x32", &f.short_encodings[..]),
        ("long_x1", &f.long_encodings[..1]),
        ("long_x32", &f.long_encodings[..]),
    ] {
        group.bench_with_input(BenchmarkId::new("cpu", name), encodings, |b, encs| {
            b.iter(|| f.backend.embed_batch(black_box(encs)).expect("embed"));
        });
    }

    group.finish();
}

// --- Tokenization benchmarks ---

fn bench_tokenize(c: &mut Criterion) {
    let f = Fixtures::load();
    let mut group = c.benchmark_group("tokenize");

    let short = "fn hello() { println!(\"world\"); }";
    let medium = (0..10)
        .map(|i| format!("let x_{i} = compute(input_{i});\n"))
        .collect::<String>();
    let long = &f.rust_source;

    for (name, text) in [
        ("short_34b", short),
        ("medium_300b", medium.as_str()),
        ("long_5kb", long.as_str()),
    ] {
        group.bench_with_input(BenchmarkId::new("encode", name), text, |b, text| {
            b.iter(|| f.tokenizer.encode(black_box(text), true).expect("tokenize"));
        });
    }

    group.finish();
}

// --- Chunking benchmarks ---

fn bench_chunk_file(c: &mut Criterion) {
    let f = Fixtures::load();
    let lang = ripvec_core::languages::config_for_extension("rs").expect("rust config");
    let chunk_cfg = ChunkConfig::default();
    let path = Path::new("bench_sample.rs");

    c.bench_function("chunk_file/rust_50fn", |b| {
        b.iter(|| {
            ripvec_core::chunk::chunk_file(
                black_box(path),
                black_box(&f.rust_source),
                black_box(&lang),
                black_box(&chunk_cfg),
            )
        });
    });
}

// --- Similarity benchmarks ---

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [384, 768, 1024] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), &(a, b), |bench, (a, b)| {
            bench.iter(|| ripvec_core::similarity::dot_product(black_box(a), black_box(b)));
        });
    }

    group.finish();
}

// --- Token length sweep: cost curve for max_tokens ---

fn bench_max_tokens_sweep(c: &mut Criterion) {
    let backend = ripvec_core::backend::load_backend(
        BackendKind::Cpu,
        MODEL_REPO,
        DeviceHint::Cpu,
        &ripvec_core::backend::InferenceOpts::default(),
    )
    .expect("backend load");
    let tokenizer = ripvec_core::tokenize::load_tokenizer(MODEL_REPO).expect("tokenizer");

    // Generate a long source text that tokenizes to 512+ tokens
    let long_source = (0..100)
        .map(|i| {
            format!(
                "fn process_{i}(data: &[u8], offset: usize) -> Result<Vec<u8>, Error> {{\n\
                     let chunk = &data[offset..offset + {i}];\n\
                     Ok(chunk.to_vec())\n\
                 }}\n"
            )
        })
        .collect::<String>();

    // Pre-tokenize at full length
    let full_enc = tokenizer
        .encode(long_source.as_str(), true)
        .expect("tokenize");
    let full_ids = full_enc.get_ids();

    let mut group = c.benchmark_group("max_tokens_sweep");
    group.sample_size(10);

    for max_tokens in [32, 64, 128, 192, 256, 384, 512] {
        let len = full_ids.len().min(max_tokens);

        // Build a batch of 32 encodings truncated to this length
        let batch: Vec<Encoding> = (0..DEFAULT_BATCH_SIZE)
            .map(|_| Encoding {
                input_ids: full_ids[..len].iter().map(|&id| i64::from(id)).collect(),
                attention_mask: vec![1i64; len],
                token_type_ids: vec![0i64; len],
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch32", max_tokens),
            &batch,
            |b, batch| {
                b.iter(|| backend.embed_batch(black_box(batch)).expect("embed"));
            },
        );
    }

    group.finish();
}

// --- Full pipeline benchmark (small directory) ---

fn bench_search(c: &mut Criterion) {
    let f = Fixtures::load();
    let profiler = Profiler::noop();
    let cfg = SearchConfig::default();

    // Benchmark against the ripvec-core crate's own src/ (small, ~10 files)
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    c.bench_function("search/core_src", |b| {
        b.iter(|| {
            ripvec_core::embed::search(
                black_box(&src_dir),
                black_box("embedding model inference"),
                black_box(&[f.backend.as_ref()]),
                black_box(&f.tokenizer),
                black_box(5),
                black_box(&cfg),
                black_box(&profiler),
            )
            .expect("search")
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_embed_batch, bench_tokenize, bench_chunk_file, bench_dot_product, bench_max_tokens_sweep, bench_search
}
criterion_main!(benches);
