use jsonschema::Validator;
use serde_json::Value;
use std::path::Path;

fn load_schema() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("data/schema.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse schema JSON: {e}"))
}

fn validate_file(schema: &Validator, rel_path: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(rel_path);

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
        .expect("schema itself is invalid — check data/schema.json");

    validate_file(&schema, "data/hf_models.json");
    validate_file(&schema, "llmfit-core/data/hf_models.json");
}
