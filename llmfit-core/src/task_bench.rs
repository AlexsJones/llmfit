//! Curated per-family use-case benchmark scores (issue #150).
//!
//! Maps model families to relative task-strength scores (0-100) aggregated
//! from public leaderboards, so a strong coding model can outrank a larger
//! generalist when the user asks for a coding recommendation. The table is
//! embedded from `data/use_case_benchmarks.json` and refreshed alongside the
//! weekly model-database update; [`score`] returns `None` for models without
//! an entry, which fall back to the name heuristics in `fit.rs`.

use std::collections::HashMap;
use std::sync::OnceLock;

const TASK_BENCH_JSON: &str = include_str!("../data/use_case_benchmarks.json");

#[derive(serde::Deserialize)]
struct FamilyEntry {
    #[serde(rename = "match")]
    patterns: Vec<String>,
    scores: HashMap<String, f64>,
}

#[derive(serde::Deserialize)]
struct BenchFile {
    families: Vec<FamilyEntry>,
}

fn table() -> &'static [FamilyEntry] {
    static TABLE: OnceLock<Vec<FamilyEntry>> = OnceLock::new();
    TABLE.get_or_init(|| {
        serde_json::from_str::<BenchFile>(TASK_BENCH_JSON)
            .expect("embedded use_case_benchmarks.json is invalid")
            .families
    })
}

/// Benchmark score for a model on a task (`"coding"`, `"reasoning"`,
/// `"chat"`), or `None` if no family entry matches.
///
/// `name_lower` must already be lowercased. When several patterns match
/// (e.g. `qwen3` and `qwen3-coder`), the longest — most specific — wins.
pub fn score(name_lower: &str, task: &str) -> Option<f64> {
    let mut best: Option<(usize, f64)> = None;
    for entry in table() {
        for pattern in &entry.patterns {
            if name_lower.contains(pattern.as_str())
                && let Some(s) = entry.scores.get(task)
                && best.is_none_or(|(len, _)| pattern.len() > len)
            {
                best = Some((pattern.len(), *s));
            }
        }
    }
    best.map(|(_, s)| s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_table_parses() {
        assert!(!table().is_empty());
    }

    #[test]
    fn test_longest_pattern_wins() {
        // "qwen3.5-coder-…" matches both "qwen3.5" and "qwen3.5-coder";
        // the coder entry must win for the coding task.
        let coder = score("qwen/qwen3.5-coder-32b-instruct", "coding").unwrap();
        let base = score("qwen/qwen3.5-32b-instruct", "coding").unwrap();
        assert!(coder > base, "coder {coder} <= base {base}");
    }

    #[test]
    fn test_unknown_family_is_none() {
        assert_eq!(score("acme/customnet-7b", "coding"), None);
        assert_eq!(score("qwen/qwen3.5-32b", "nonexistent-task"), None);
    }

    #[test]
    fn test_coding_specialist_beats_generalist_at_coding_only() {
        let starcoder_code = score("bigcode/starcoder2-15b", "coding").unwrap();
        let llama_code = score("meta-llama/llama-3.3-70b", "coding").unwrap();
        assert!(starcoder_code > llama_code);

        let starcoder_chat = score("bigcode/starcoder2-15b", "chat").unwrap();
        let llama_chat = score("meta-llama/llama-3.3-70b", "chat").unwrap();
        assert!(llama_chat > starcoder_chat);
    }
}
