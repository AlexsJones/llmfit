use llmfit_core::{ModelDatabase, ModelFormat};

const CORE_ONNX_MODELS_JSON: &str = include_str!("../data/onnx_models.json");
const ROOT_ONNX_MODELS_JSON: &str = include_str!("../../data/onnx_models.json");

fn load_catalog() -> Vec<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(CORE_ONNX_MODELS_JSON)
        .expect("embedded onnx_models.json is valid JSON");
    value
        .as_array()
        .expect("embedded onnx_models.json is a JSON array")
        .clone()
}

#[test]
fn core_and_repo_root_onnx_catalogs_are_mirrored() {
    assert_eq!(
        CORE_ONNX_MODELS_JSON, ROOT_ONNX_MODELS_JSON,
        "repo-root and llmfit-core ONNX catalogs must stay in sync"
    );
}

#[test]
fn onnx_catalog_is_non_empty() {
    let models = load_catalog();
    assert!(
        !models.is_empty(),
        "embedded onnx_models.json must contain at least one model"
    );
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

#[test]
fn embedded_database_contains_onnx_catalog_models() {
    let catalog = load_catalog();
    let db = ModelDatabase::embedded();

    for entry in catalog {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .expect("catalog entry has id");
        let model = db
            .get_all_models()
            .iter()
            .find(|model| model.name == id)
            .unwrap_or_else(|| panic!("embedded database must include ONNX model {id}"));

        assert_eq!(model.format, ModelFormat::Onnx);
        assert_eq!(model.provider, id.split('/').next().unwrap_or_default());
        assert!(
            model.min_ram_gb > 0.0,
            "ONNX model {id} must have a positive RAM estimate"
        );
        assert!(
            model.recommended_ram_gb >= model.min_ram_gb,
            "ONNX model {id} recommended RAM must be at least min RAM"
        );
    }
}
