# COMPLETE EXPERIMENTAL RESULTS AUDIT
Generated: 2025-12-20 18:55:57

**NOTE: All values below are from actual executed experiments.**

## SUMMARY
- **Total experiments audited**: 170
- **Best accuracy**: 74.0%
- **Best experiment**: exp_002_supervised

**Source files are referenced for verification.**

## Source: results_gpu

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_008_lr_2e-4 | rl | 53.7% | 73.8% | 63.2% | 58.0% | 20.7% |
| exp_009_lr_5e-4 | rl | 44.0% | 30.2% | 61.1% | 58.0% | 27.7% |
| exp_023_reward_progressive_slow | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_024_qtype_color_only | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_025_qtype_shape_only | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_026_qtype_count_only | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_027_qtype_spatial_only | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_028_qtype_color_shape | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_029_qtype_color_count | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |
| exp_002_supervised | supervised | 33.7% | 40.1% | 26.3% | 58.0% | 11.3% |
| exp_006_lr_5e-5 | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_016_reward_length_penalty | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_017_reward_combined | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_018_reward_progressive | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_019_reward_exact_strict | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_020_reward_partial_strict | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_021_reward_combined_v2 | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_022_reward_progressive_fast | rl | 32.4% | 33.3% | 30.4% | 58.0% | 9.0% |
| exp_003_rl_baseline | rl | 31.7% | 33.3% | 25.9% | 58.0% | 10.5% |
| exp_007_lr_1e-4 | rl | 31.7% | 33.3% | 25.9% | 58.0% | 10.5% |
| exp_010_lr_1e-3 | rl | 29.3% | 31.0% | 20.6% | 58.0% | 8.6% |
| exp_014_reward_exact_match | rl | 29.3% | 31.0% | 20.6% | 58.0% | 8.6% |
| exp_015_reward_partial_match | rl | 29.3% | 31.0% | 20.6% | 58.0% | 8.6% |
| exp_011_lr_2e-3 | rl | 20.7% | 25.8% | 0.0% | 58.0% | 0.0% |
| exp_004_lr_1e-5 | rl | 18.8% | 4.8% | 10.5% | 58.0% | 3.1% |
| exp_005_lr_2e-5 | rl | 14.2% | 0.0% | 0.0% | 58.0% | 0.0% |
| exp_012_lr_5e-3 | rl | 14.2% | 0.0% | 0.0% | 58.0% | 0.0% |
| exp_013_lr_1e-2 | rl | 14.2% | 0.0% | 0.0% | 58.0% | 0.0% |
| exp_001_frozen | frozen | 13.2% | 19.8% | 11.3% | 16.7% | 5.1% |

## Source: results_gpu_batch

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_029_qtype_color_count | rl | 43.1% | 54.0% | 26.3% | 58.0% | 34.4% |

## Source: results_fixed

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_002_supervised | supervised | 74.0% | 77.4% | 75.7% | 82.0% | 61.3% |
| exp_003_rl_baseline | rl | 47.6% | 71.8% | 20.6% | 58.0% | 39.8% |
| exp_007_lr_1e-4 | rl | 45.2% | 53.6% | 28.3% | 58.0% | 41.0% |
| exp_006_lr_5e-5 | rl | 41.0% | 44.0% | 27.1% | 58.0% | 35.2% |
| exp_005_lr_2e-5 | rl | 37.0% | 30.2% | 25.1% | 58.0% | 35.2% |
| exp_004_lr_1e-5 | rl | 29.4% | 0.0% | 25.1% | 58.0% | 35.2% |
| exp_001_frozen | frozen | 0.2% | 0.0% | 0.0% | 0.0% | 0.8% |

## Source: results_fixed_batch

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_002_supervised | supervised | 74.0% | 77.4% | 75.7% | 82.0% | 61.3% |
| exp_003_rl_baseline | rl | 47.6% | 71.8% | 20.6% | 58.0% | 39.8% |
| exp_007_lr_1e-4 | rl | 45.2% | 53.6% | 28.3% | 58.0% | 41.0% |
| exp_006_lr_5e-5 | rl | 41.0% | 44.0% | 27.1% | 58.0% | 35.2% |
| exp_005_lr_2e-5 | rl | 37.0% | 30.2% | 25.1% | 58.0% | 35.2% |
| exp_004_lr_1e-5 | rl | 29.4% | 0.0% | 25.1% | 58.0% | 35.2% |
| exp_001_frozen | frozen | 0.2% | 0.0% | 0.0% | 0.0% | 0.8% |

## Source: results_high_acc

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_032_qtype_no_spatial | rl | 61.9% | 75.0% | 61.9% | 58.0% | 52.7% |
| exp_033_qtype_all_types | rl | 61.9% | 75.0% | 61.9% | 58.0% | 52.7% |
| exp_030_qtype_shape_count | rl | 51.7% | 59.9% | 45.7% | 58.0% | 43.4% |
| exp_031_qtype_color_spatial | rl | 51.7% | 59.9% | 45.7% | 58.0% | 43.4% |

## Source: results_high_acc_batch

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_032_qtype_no_spatial | rl | 61.9% | 75.0% | 61.9% | 58.0% | 52.7% |
| exp_033_qtype_all_types | rl | 61.9% | 75.0% | 61.9% | 58.0% | 52.7% |
| exp_030_qtype_shape_count | rl | 51.7% | 59.9% | 45.7% | 58.0% | 43.4% |
| exp_031_qtype_color_spatial | rl | 51.7% | 59.9% | 45.7% | 58.0% | 43.4% |

## Source: results

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| exp_002_supervised | supervised | 1.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_001_frozen | frozen | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_003_rl_baseline | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_004_lr_1e-5 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_005_lr_2e-5 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_006_lr_5e-5 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_007_lr_1e-4 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_008_lr_2e-4 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_009_lr_5e-4 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_010_lr_1e-3 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_011_lr_2e-3 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_012_lr_5e-3 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_013_lr_1e-2 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_017_reward_combined | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_018_reward_progressive | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_021_reward_combined_v2 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_022_reward_progressive_fast | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_023_reward_progressive_slow | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_024_qtype_color_only | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_025_qtype_shape_only | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_026_qtype_count_only | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_027_qtype_spatial_only | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_028_qtype_color_shape | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_029_qtype_color_count | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_030_qtype_shape_count | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_031_qtype_color_spatial | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_032_qtype_no_spatial | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_033_qtype_all_types | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_034_baseline_none | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_035_baseline_mavg_0.9 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_036_baseline_mavg_0.95 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_037_baseline_mavg_0.99 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_038_baseline_mavg_0.999 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_039_baseline_learned | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_040_baseline_mavg_0.8 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_041_baseline_mavg_0.7 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_042_baseline_mavg_0.5 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_043_baseline_mavg_0.0 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_044_temp_0_1 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_045_temp_0_3 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_046_temp_0_5 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_047_temp_0_7 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_048_temp_1_0 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_049_temp_1_5 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_050_temp_2_0 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_051_temp_3_0 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_052_entropy_0_0 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_053_entropy_0_001 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_054_entropy_0_01 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_055_entropy_0_05 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_056_entropy_0_1 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_057_batch_4 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_058_batch_8 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_059_batch_16 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_060_batch_32 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| exp_061_batch_64 | rl | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Source: results_batch

| Experiment | Method | Accuracy | Shape | Color | Count | Spatial |
|------------|--------|----------|-------|-------|-------|---------|
| unknown | unknown | 1.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| unknown | unknown | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Source: high_accuracy_model

- **Accuracy**: 68.7%
- **Method**: supervised
- **Per-type**: Shape=79.4%, Color=71.3%, Count=62.0%, Spatial=62.1%
