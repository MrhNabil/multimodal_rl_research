#!/usr/bin/env python
"""
Collect and summarize ALL existing experimental results.
This script ONLY reads actual logged data - no fabrication.
"""

import os
import json
import glob
from datetime import datetime

# All result directories
RESULT_DIRS = [
    r"d:\multimodal_rl_research\experiments\results_gpu",
    r"d:\multimodal_rl_research\experiments\results_fixed",
    r"d:\multimodal_rl_research\experiments\results_high_acc",
    r"d:\multimodal_rl_research\experiments\results",
]

OUTPUT_FILE = r"d:\multimodal_rl_research\experiments\COMPLETE_RESULTS_AUDIT.json"
OUTPUT_TABLE = r"d:\multimodal_rl_research\experiments\COMPLETE_RESULTS_TABLE.md"

def collect_results():
    """Collect all results from all directories."""
    all_results = {}
    
    for result_dir in RESULT_DIRS:
        if not os.path.exists(result_dir):
            print(f"Directory not found: {result_dir}")
            continue
            
        dir_name = os.path.basename(result_dir)
        all_results[dir_name] = []
        
        # Check for batch_summary.json first
        batch_file = os.path.join(result_dir, "batch_summary.json")
        if os.path.exists(batch_file):
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
                print(f"Found batch_summary in {dir_name}: {len(batch_data)} experiments")
                all_results[dir_name + "_batch"] = batch_data
        
        # Also collect individual final_results.json
        for exp_dir in sorted(glob.glob(os.path.join(result_dir, "exp_*"))):
            exp_name = os.path.basename(exp_dir)
            
            # Check for final_results.json
            results_file = os.path.join(exp_dir, "final_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    data['experiment_name'] = exp_name
                    data['source_dir'] = dir_name
                    all_results[dir_name].append(data)
    
    # Check high_accuracy special result
    ha_file = r"d:\multimodal_rl_research\experiments\high_accuracy\results.json"
    if os.path.exists(ha_file):
        with open(ha_file, 'r') as f:
            all_results['high_accuracy_model'] = json.load(f)
            print("Found high_accuracy model result")
    
    return all_results

def generate_table(all_results):
    """Generate markdown table from results."""
    lines = []
    lines.append("# COMPLETE EXPERIMENTAL RESULTS AUDIT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("**NOTE: All values below are from actual executed experiments.**")
    lines.append("**Source files are referenced for verification.**")
    lines.append("")
    
    # Summary statistics
    total_experiments = 0
    best_accuracy = 0
    best_experiment = ""
    
    for source, results in all_results.items():
        if isinstance(results, list) and len(results) > 0:
            lines.append(f"## Source: {source}")
            lines.append("")
            lines.append("| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |")
            lines.append("|------------|--------|----------|-------|-------|-------|---------|")
            
            for r in sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True):
                exp_name = r.get('experiment_name', r.get('name', 'unknown'))
                method = r.get('method', 'unknown')
                acc = r.get('accuracy', 0) * 100
                
                # Per-type accuracy
                per_type = r.get('per_type_accuracy', r.get('per_type', r.get('by_type', {})))
                shape = per_type.get('shape', 0) * 100 if per_type else 0
                color = per_type.get('color', 0) * 100 if per_type else 0
                count = per_type.get('count', 0) * 100 if per_type else 0
                spatial = per_type.get('spatial', 0) * 100 if per_type else 0
                
                lines.append(f"| {exp_name} | {method} | {acc:.1f}% | {shape:.1f}% | {color:.1f}% | {count:.1f}% | {spatial:.1f}% |")
                
                total_experiments += 1
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_experiment = exp_name
            
            lines.append("")
        
        elif isinstance(results, dict) and 'test_accuracy' in results:
            lines.append(f"## Source: {source}")
            lines.append("")
            acc = results.get('test_accuracy', 0) * 100
            method = results.get('method', 'unknown')
            by_type = results.get('by_type', {})
            
            lines.append(f"- **Accuracy**: {acc:.1f}%")
            lines.append(f"- **Method**: {method}")
            lines.append(f"- **Per-type**: Shape={by_type.get('shape',0)*100:.1f}%, Color={by_type.get('color',0)*100:.1f}%, Count={by_type.get('count',0)*100:.1f}%, Spatial={by_type.get('spatial',0)*100:.1f}%")
            lines.append("")
            
            total_experiments += 1
            if acc > best_accuracy:
                best_accuracy = acc
                best_experiment = source
    
    # Summary
    lines.insert(4, f"")
    lines.insert(5, f"## SUMMARY")
    lines.insert(6, f"- **Total experiments audited**: {total_experiments}")
    lines.insert(7, f"- **Best accuracy**: {best_accuracy:.1f}%")
    lines.insert(8, f"- **Best experiment**: {best_experiment}")
    lines.insert(9, f"")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print("Auditing all experimental results...")
    print("=" * 60)
    
    all_results = collect_results()
    
    # Save JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_FILE}")
    
    # Save table
    table = generate_table(all_results)
    with open(OUTPUT_TABLE, 'w') as f:
        f.write(table)
    print(f"Table saved: {OUTPUT_TABLE}")
    
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
