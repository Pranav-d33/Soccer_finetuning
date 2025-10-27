"""
Dataset Validation & Analysis Script
Validates the generated instruction datasets and provides quality metrics
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
import statistics

def load_jsonl(file_path: str):
    """Load JSONL file into a list of dictionaries."""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Error on line {line_num}: {e}")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    
    return samples

def analyze_dataset(file_path: str, name: str = "Dataset"):
    """Analyze a dataset file and print statistics."""
    print(f"\n{'='*80}")
    print(f"📊 ANALYZING: {name}")
    print(f"{'='*80}")
    
    samples = load_jsonl(file_path)
    if not samples:
        return
    
    print(f"\n📈 Basic Statistics:")
    print(f"   Total Samples: {len(samples):,}")
    
    # Extract metrics
    instruction_lengths = [len(s.get('instruction', '')) for s in samples]
    input_lengths = [len(s.get('input', '')) for s in samples]
    output_lengths = [len(s.get('output', '')) for s in samples]
    
    print(f"\n📝 Text Lengths:")
    print(f"   Instruction:")
    print(f"      Min: {min(instruction_lengths)} | Max: {max(instruction_lengths)} | Avg: {statistics.mean(instruction_lengths):.0f}")
    print(f"   Input:")
    print(f"      Min: {min(input_lengths)} | Max: {max(input_lengths)} | Avg: {statistics.mean(input_lengths):.0f}")
    print(f"   Output:")
    print(f"      Min: {min(output_lengths)} | Max: {max(output_lengths)} | Avg: {statistics.mean(output_lengths):.0f}")
    
    # Instruction types
    types = [s.get('metadata', {}).get('type', 'unknown') for s in samples]
    type_counts = Counter(types)
    
    print(f"\n🎯 Instruction Types:")
    for itype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(samples)) * 100
        bar = "█" * int(percentage / 5)
        print(f"   {itype:25} {count:6,} ({percentage:5.1f}%) {bar}")
    
    # Team diversity
    home_teams = [s.get('metadata', {}).get('home_team') for s in samples]
    away_teams = [s.get('metadata', {}).get('away_team') for s in samples]
    
    unique_home = len(set(home_teams))
    unique_away = len(set(away_teams))
    
    print(f"\n🏆 Team Diversity:")
    print(f"   Unique Home Teams: {unique_home}")
    print(f"   Unique Away Teams: {unique_away}")
    
    # Match diversity
    match_ids = [s.get('metadata', {}).get('match_id') for s in samples]
    unique_matches = len(set(match_ids))
    print(f"   Unique Matches: {unique_matches}")
    
    # Quality checks
    print(f"\n✅ Quality Checks:")
    
    # Check for required fields
    missing_fields = {'instruction': 0, 'input': 0, 'output': 0}
    for sample in samples:
        for field in missing_fields:
            if not sample.get(field):
                missing_fields[field] += 1
    
    for field, count in missing_fields.items():
        status = "✓" if count == 0 else "✗"
        print(f"   {status} Missing '{field}': {count}")
    
    # Check for reasonable lengths
    short_outputs = sum(1 for length in output_lengths if length < 50)
    long_outputs = sum(1 for length in output_lengths if length > 2000)
    
    print(f"   ⚠️  Very short outputs (<50 chars): {short_outputs}")
    print(f"   ⚠️  Very long outputs (>2000 chars): {long_outputs}")
    
    # Sample display
    print(f"\n📋 Sample Examples:")
    for i in range(min(2, len(samples))):
        sample = samples[i]
        print(f"\n   Example {i+1}:")
        print(f"   Type: {sample.get('metadata', {}).get('type', 'unknown')}")
        print(f"   Instruction: {sample.get('instruction', '')[:80]}...")
        print(f"   Output: {sample.get('output', '')[:100]}...")
    
    return samples

def compare_datasets(train_file: str, val_file: str):
    """Compare training and validation datasets."""
    print(f"\n{'='*80}")
    print(f"📊 COMPARING DATASETS")
    print(f"{'='*80}")
    
    train_samples = load_jsonl(train_file)
    val_samples = load_jsonl(val_file)
    
    if not train_samples or not val_samples:
        print("❌ Could not load both files")
        return
    
    train_types = Counter([s.get('metadata', {}).get('type') for s in train_samples])
    val_types = Counter([s.get('metadata', {}).get('type') for s in val_samples])
    
    print(f"\n📊 Distribution Comparison:")
    print(f"{'Type':<25} {'Train':<15} {'Val':<15} {'Ratio':<10}")
    print("-" * 65)
    
    for itype in sorted(set(list(train_types.keys()) + list(val_types.keys()))):
        train_count = train_types.get(itype, 0)
        val_count = val_types.get(itype, 0)
        train_pct = (train_count / len(train_samples)) * 100 if train_samples else 0
        val_pct = (val_count / len(val_samples)) * 100 if val_samples else 0
        
        print(f"{itype:<25} {train_count:>6,} ({train_pct:>5.1f}%)  "
              f"{val_count:>6,} ({val_pct:>5.1f}%)  "
              f"{train_pct/val_pct:>5.2f}x")
    
    print(f"\n📈 Split Statistics:")
    total = len(train_samples) + len(val_samples)
    train_ratio = (len(train_samples) / total) * 100
    val_ratio = (len(val_samples) / total) * 100
    
    print(f"   Training Samples: {len(train_samples):,} ({train_ratio:.1f}%)")
    print(f"   Validation Samples: {len(val_samples):,} ({val_ratio:.1f}%)")
    print(f"   Total Samples: {total:,}")

def main():
    """Run all analyses."""
    base_path = Path(".")
    
    train_file = base_path / "football_train.jsonl"
    val_file = base_path / "football_val.jsonl"
    
    if not train_file.exists() or not val_file.exists():
        print("❌ Dataset files not found!")
        print("   Please run: python generate_dataset.py")
        return
    
    print("\n" + "="*80)
    print("🔍 DATASET VALIDATION & ANALYSIS")
    print("="*80)
    
    # Analyze individual datasets
    train_samples = analyze_dataset(str(train_file), "Training Dataset")
    val_samples = analyze_dataset(str(val_file), "Validation Dataset")
    
    # Compare datasets
    compare_datasets(str(train_file), str(val_file))
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
