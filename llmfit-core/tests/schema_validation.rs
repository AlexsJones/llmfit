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
    let required = [(
        "QwerkyAI/Qwerky-Optimized-Llama3.2-Mamba-0.2-3B-Instruct",
        24u64,
        8u64,
    )];
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
