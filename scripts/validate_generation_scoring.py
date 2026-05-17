#!/usr/bin/env python3
"""
Validate generation-aware quality scoring on the scraped model database.

Loads hf_models.json and applies the same generation parsing logic as the Rust
code to demonstrate that newer model generations are scored appropriately.
"""

import json
import re
import sys
from collections import defaultdict


def parse_generation(architecture: str | None, name: str) -> float | None:
    """Mirror of models::parse_generation in Rust."""
    if architecture:
        arch = architecture.lower()

        # DeepSeek
        if arch.startswith("deepseek"):
            if "v4" in arch:
                return 4.0
            elif "v3" in arch:
                return 3.0
            elif "v2" in arch:
                return 2.0
            return 1.0

        # Qwen
        if arch.startswith("qwen"):
            suffix = arch[len("qwen"):]
            if suffix.startswith("3_5") or suffix.startswith("3.5"):
                return 3.5
            if suffix.startswith("3_next") or suffix.startswith("3next"):
                return 3.8
            if suffix.startswith("3"):
                return 3.0
            if suffix.startswith("2"):
                return 2.0
            if suffix.startswith("1"):
                return 1.0
            return 1.0

        # Llama
        if arch.startswith("llama"):
            suffix = arch[len("llama"):]
            if suffix.startswith("4"):
                return 4.0
            # fall through to name

        # Gemma
        if arch.startswith("gemma"):
            suffix = arch[len("gemma"):]
            if suffix.startswith("4"):
                return 4.0
            if suffix.startswith("3"):
                return 3.0
            if suffix.startswith("2"):
                return 2.0
            return 1.0

        # Phi
        if arch.startswith("phi"):
            suffix = arch[len("phi"):]
            if suffix.startswith("4"):
                return 4.0
            if suffix.startswith("3") or suffix.startswith("moe"):
                return 3.0
            if suffix.startswith("2"):
                return 2.0
            return 1.0

        # Mistral/Mixtral
        if arch.startswith("mistral") or arch.startswith("mixtral"):
            return 1.0

        # Cohere
        if arch.startswith("cohere"):
            suffix = arch[len("cohere"):]
            if suffix.startswith("2"):
                return 2.0
            return 1.0

        # Falcon
        if arch.startswith("falcon"):
            suffix = arch[len("falcon"):]
            if suffix.startswith("3"):
                return 3.0
            return 1.0

        # Granite
        if arch.startswith("granite"):
            suffix = arch[len("granite"):]
            if suffix.startswith("4"):
                return 4.0
            return 1.0

    # Fallback: name-based
    name_lower = name.lower()

    if "qwen3.6" in name_lower or "qwen3_6" in name_lower:
        return 3.6
    if "qwen3.5" in name_lower or "qwen3_5" in name_lower:
        return 3.5
    if "qwen3" in name_lower:
        return 3.0
    if "qwen2.5" in name_lower or "qwen2_5" in name_lower:
        return 2.5
    if "qwen2" in name_lower:
        return 2.0

    if "llama-4" in name_lower or "llama4" in name_lower:
        return 4.0
    if "llama-3.3" in name_lower or "llama3.3" in name_lower:
        return 3.3
    if "llama-3.2" in name_lower or "llama3.2" in name_lower:
        return 3.2
    if "llama-3.1" in name_lower or "llama3.1" in name_lower:
        return 3.1
    if "llama-3" in name_lower or "llama3" in name_lower:
        return 3.0
    if "llama-2" in name_lower or "llama2" in name_lower:
        return 2.0

    if "gemma-4" in name_lower or "gemma4" in name_lower:
        return 4.0
    if "gemma-3" in name_lower or "gemma3" in name_lower:
        return 3.0
    if "gemma-2" in name_lower or "gemma2" in name_lower:
        return 2.0

    if "deepseek-v4" in name_lower:
        return 4.0
    if "deepseek-v3" in name_lower:
        return 3.0
    if "deepseek-v2" in name_lower:
        return 2.0

    if "phi-4" in name_lower or "phi4" in name_lower:
        return 4.0
    if "phi-3" in name_lower or "phi3" in name_lower:
        return 3.0

    return None


def generation_bonus(architecture: str | None, name: str) -> float:
    gen = parse_generation(architecture, name)
    if gen is None:
        return 0.0
    return min((gen - 1.0) * 3.0, 9.0)


def params_b(model: dict) -> float:
    """Extract parameter count in billions."""
    if model.get("parameters_raw"):
        return model["parameters_raw"] / 1e9
    pc = model.get("parameter_count", "")
    m = re.search(r"([\d.]+)\s*[Bb]", pc)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*[Mm]", pc)
    if m:
        return float(m.group(1)) / 1000
    return 0.0


def quality_score_old(model: dict) -> float:
    """Old scoring without generation bonus."""
    params = params_b(model)
    if params < 1.0:
        base = 30.0
    elif params < 3.0:
        base = 45.0
    elif params < 7.0:
        base = 60.0
    elif params < 10.0:
        base = 75.0
    elif params < 20.0:
        base = 82.0
    elif params < 40.0:
        base = 89.0
    else:
        base = 95.0

    name_lower = model["name"].lower()
    if "qwen" in name_lower:
        family_bump = 2.0
    elif "deepseek" in name_lower:
        family_bump = 3.0
    elif "llama" in name_lower:
        family_bump = 2.0
    elif "mistral" in name_lower or "mixtral" in name_lower:
        family_bump = 1.0
    elif "gemma" in name_lower:
        family_bump = 1.0
    else:
        family_bump = 0.0

    return min(base + family_bump - 5.0, 100.0)  # -5 for Q4_K_M penalty


def quality_score_new(model: dict) -> float:
    """New scoring with generation bonus."""
    params = params_b(model)
    if params < 1.0:
        base = 30.0
    elif params < 3.0:
        base = 45.0
    elif params < 7.0:
        base = 60.0
    elif params < 10.0:
        base = 75.0
    elif params < 20.0:
        base = 82.0
    elif params < 40.0:
        base = 89.0
    else:
        base = 95.0

    name_lower = model["name"].lower()
    if "qwen" in name_lower:
        family_bump = 2.0
    elif "deepseek" in name_lower:
        family_bump = 3.0
    elif "llama" in name_lower:
        family_bump = 2.0
    elif "mistral" in name_lower or "mixtral" in name_lower:
        family_bump = 1.0
    elif "gemma" in name_lower:
        family_bump = 1.0
    else:
        family_bump = 0.0

    gen_bonus = generation_bonus(model.get("architecture"), model["name"])

    return min(base + family_bump + gen_bonus - 5.0, 100.0)  # -5 for Q4_K_M penalty


def main():
    data_path = "data/hf_models.json"
    with open(data_path) as f:
        models = json.load(f)

    total = len(models)
    print(f"Loaded {total} models from {data_path}\n")

    # Count models with parseable generation
    gen_counts = defaultdict(int)
    family_gens = defaultdict(set)
    no_gen = 0
    has_gen = 0

    for m in models:
        gen = parse_generation(m.get("architecture"), m["name"])
        if gen is not None:
            has_gen += 1
            gen_counts[gen] += 1
            # Extract family
            name_lower = m["name"].lower()
            for fam in ["qwen", "llama", "deepseek", "gemma", "phi", "mistral", "falcon", "granite"]:
                if fam in name_lower:
                    family_gens[fam].add(gen)
                    break
        else:
            no_gen += 1

    print(f"Generation coverage: {has_gen}/{total} ({100*has_gen/total:.1f}%)")
    print(f"No generation info:  {no_gen}/{total} ({100*no_gen/total:.1f}%)\n")

    print("Generation distribution:")
    for gen in sorted(gen_counts.keys()):
        print(f"  Gen {gen:>4.1f}: {gen_counts[gen]:>4} models")

    print(f"\nFamilies with multiple generations ({len([f for f in family_gens if len(family_gens[f]) > 1])}):")
    for fam in sorted(family_gens.keys()):
        gens = sorted(family_gens[fam])
        if len(gens) > 1:
            print(f"  {fam:>10}: gens {', '.join(f'{g:.1f}' for g in gens)}")

    # Show specific ranking improvements
    print("\n" + "=" * 70)
    print("KEY COMPARISONS: Generation-aware scoring fixes")
    print("=" * 70)

    comparisons = [
        ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen2.5-72B-Instruct"),
        ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"),
        ("google/gemma-4-E2B-it", "google/gemma-2-2b-it"),
        ("google/gemma-3-27b-it", "google/gemma-2-27b-it"),
    ]

    model_lookup = {m["name"]: m for m in models}

    for newer_name, older_name in comparisons:
        newer = model_lookup.get(newer_name)
        older = model_lookup.get(older_name)
        if not newer or not older:
            print(f"\n  SKIP: {newer_name} or {older_name} not found in database")
            continue

        old_newer = quality_score_old(newer)
        old_older = quality_score_old(older)
        new_newer = quality_score_new(newer)
        new_older = quality_score_new(older)

        newer_gen = parse_generation(newer.get("architecture"), newer["name"])
        older_gen = parse_generation(older.get("architecture"), older["name"])

        print(f"\n  {newer_name} (gen {newer_gen}, {params_b(newer):.1f}B)")
        print(f"    vs {older_name} (gen {older_gen}, {params_b(older):.1f}B)")
        print(f"    OLD: {old_newer:.1f} vs {old_older:.1f} (gap: {old_older - old_newer:+.1f})")
        print(f"    NEW: {new_newer:.1f} vs {new_older:.1f} (gap: {new_older - new_newer:+.1f})")

        if old_older - old_newer > new_older - new_newer:
            print(f"    ✓ Gap narrowed by {(old_older - old_newer) - (new_older - new_newer):.1f} points")
        if new_newer > new_older:
            print(f"    ✓ RANKING FIXED: newer model now scores higher")

    # Show top models by new score in each family
    print("\n" + "=" * 70)
    print("TOP 5 BY QUALITY SCORE (new vs old) — selected families")
    print("=" * 70)

    for family in ["qwen", "llama", "gemma", "deepseek"]:
        family_models = [m for m in models if family in m["name"].lower()]
        # Sort by new score, take top 5
        family_models.sort(key=lambda m: quality_score_new(m), reverse=True)
        print(f"\n  {family.upper()} (top 5):")
        for m in family_models[:5]:
            gen = parse_generation(m.get("architecture"), m["name"])
            old = quality_score_old(m)
            new = quality_score_new(m)
            print(f"    {m['name'][:50]:<50} gen={gen}  old={old:5.1f}  new={new:5.1f}  Δ={new-old:+.1f}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    score_changes = []
    for m in models:
        old = quality_score_old(m)
        new = quality_score_new(m)
        if new != old:
            score_changes.append((m["name"], old, new, new - old))

    print(f"\n  Models with score changes: {len(score_changes)}/{total} ({100*len(score_changes)/total:.1f}%)")
    print(f"  Models unchanged:          {total - len(score_changes)}/{total}")

    if score_changes:
        deltas = [x[3] for x in score_changes]
        print(f"  Average score increase:    {sum(deltas)/len(deltas):+.2f}")
        print(f"  Max score increase:        {max(deltas):+.1f}")
        print(f"  Min score increase:        {min(deltas):+.1f}")


if __name__ == "__main__":
    main()
