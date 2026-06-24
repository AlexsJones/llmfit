use std::fs;

fn load_catalog() -> Vec<serde_json::Value> {
    let raw = fs::read_to_string("../data/onnx_models.json")
        .expect("failed to read data/onnx_models.json");
    let value: serde_json::Value =
        serde_json::from_str(&raw).expect("data/onnx_models.json is not valid JSON");
    value
        .as_array()
        .expect("data/onnx_models.json must be a JSON array")
        .clone()
}

#[test]
fn onnx_catalog_is_non_empty() {
    let models = load_catalog();
    assert!(!models.is_empty(), "onnx_models.json must contain at least one model");
}

#[test]
fn every_model_is_marked_as_onnx() {
    for model in load_catalog() {
        assert_eq!(
            model.get("format").and_then(|v| v.as_str()),
            Some("onnx"),
            "model {:?} must set format = \"onnx\"",
            model.get("id")
        );
    }
}

#[test]
fn every_model_has_positive_quantization_sizes() {
    for model in load_catalog() {
        let onnx_files = model
            .get("onnx_files")
            .and_then(|v| v.as_object())
            .unwrap_or_else(|| panic!("model {:?} must include onnx_files", model.get("id")));
        assert!(
            !onnx_files.is_empty(),
            "model {:?} must list at least one quantization",
            model.get("id")
        );
        for (quant, size) in onnx_files {
            let bytes = size.as_u64().unwrap_or(0);
            assert!(
                bytes > 0,
                "quantization {} for model {:?} must have a positive byte size",
                quant,
                model.get("id")
            );
        }
    }
}

#[test]
fn model_ids_are_unique() {
    let models = load_catalog();
    let mut ids: Vec<&str> = models
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(ids.len(), models.len(), "every model must have a string id");
    let total = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), total, "model ids must be unique");
}
