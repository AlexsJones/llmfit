use jsonschema::Validator;
use serde_json::Value;
use std::path::Path;

fn load_schema() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/schema.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse schema JSON: {e}"))
}

fn validate_file(schema: &Validator, rel_path: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel_path);

    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let data: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()));

    let errors: Vec<String> = schema
        .iter_errors(&data)
        .take(30)
        .map(|e| format!("  [{}]  {}", e.instance_path(), e))
        .collect();

    assert!(
        errors.is_empty(),
        "{rel_path}: {} schema violation(s):\n{}",
        errors.len(),
        errors.join("\n")
    );

    let count = data.as_array().map(|a| a.len()).unwrap_or(0);
    println!("✓ {rel_path}  ({count} models)");
}

#[test]
fn hf_models_match_schema() {
    let schema_value = load_schema();
    let schema = jsonschema::validator_for(&schema_value)
        .expect("schema itself is invalid — check llmfit-core/data/schema.json");

    validate_file(&schema, "data/hf_models.json");
}

#[test]
fn hf_models_keep_known_hybrid_attention_heads() {
    // Regression for the 2026-08-28 weekly scrape: a missed config.json fetch
    // nullified architecture fields, and "mamba" in the name then erased KV.
    // Weekly CI runs `hf_models_*`; keep this name so it gates the refresh PR.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/hf_models.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let data: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()));
    let models = data.as_array().expect("hf_models.json is an array");
    let required = [
        (
            "QwerkyAI/Qwerky-Optimized-Llama3.2-Mamba-0.2-3B-Instruct",
            24u64,
            8u64,
        ),
        // Backfilled from config.json after v1.1.19 cluster mode could not
        // check their tensor-parallel splits (gated 405B: public mirror config).
        ("meta-llama/Llama-3.1-405B", 128, 8),
        ("meta-llama/Llama-3.1-405B-Instruct", 128, 8),
        ("meta-llama/Llama-3.1-405B-FP8", 128, 8),
        ("MiniMaxAI/MiniMax-M1-80k", 64, 8),
    ];
    for (name, attn, kv) in required {
        let model = models
            .iter()
            .find(|m| m.get("name").and_then(|v| v.as_str()) == Some(name))
            .unwrap_or_else(|| panic!("missing {name} in embedded catalog"));
        assert_eq!(
            model.get("num_attention_heads").and_then(|v| v.as_u64()),
            Some(attn),
            "{name} lost num_attention_heads"
        );
        assert_eq!(
            model.get("num_key_value_heads").and_then(|v| v.as_u64()),
            Some(kv),
            "{name} lost num_key_value_heads"
        );
    }
}

#[test]
fn hf_models_backfilled_heads_make_cluster_tp_checks_real() {
    // v1.1.19 on a 3-node cluster: Llama-3.1-405B had null heads, so the TP=3
    // split was only "unverified". With real heads (128/8) it is refused, and
    // MiniMax-M1 keeps its MoE shape instead of being sized as dense.
    let db = llmfit_core::ModelDatabase::new();
    let find = |name: &str| {
        db.get_all_models()
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    let llama = find("meta-llama/Llama-3.1-405B-Instruct");
    assert!(
        !llama.supports_tp(3),
        "128 heads / 8 KV heads cannot split 3 ways"
    );
    assert!(llama.supports_tp(2) && llama.supports_tp(4) && llama.supports_tp(8));
    assert_eq!(llama.context_length, 131072);

    let minimax = find("MiniMaxAI/MiniMax-M1-80k");
    assert!(minimax.is_moe, "MiniMax-M1 is a 32-expert MoE");
    assert_eq!(minimax.num_experts, Some(32));
    assert_eq!(minimax.active_experts, Some(2));
    assert_eq!(minimax.context_length, 1_000_000);
}
