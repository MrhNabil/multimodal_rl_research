# COMPLETE RUNBOOK: Multimodal RL Research Project

## STEP 0: Environment Setup ✅ (ALREADY DONE)

Your environment is already set up. You have:
- Python virtual environment at `.venv`
- Dependencies installed via `pip install -r requirements.txt`
- Models downloaded (CLIP, T5-small) at `models/pretrained/`

**Verify setup:**
```powershell
python -c "import torch; import transformers; import open_clip; print('OK')"
```

---

## STEP 1: Dataset Status ✅ (ALREADY DONE)

Your synthetic dataset is already generated at `data/generated/`:
- **Train**: 5,000 samples
- **Validation**: 1,000 samples  
- **Test**: 1,000 samples
- **Question types**: color, shape, count, spatial

**To regenerate (optional):**
```powershell
python scripts/prepare_data.py --num_train 5000 --num_val 1000 --num_test 1000
```

---

## STEP 2: Experiment Configuration Status ✅ (ALREADY DONE)

All 61 experiment configs exist in `config/experiments/`:

| Range | Variable | Count |
|-------|----------|-------|
| 001-003 | Baseline methods (frozen/supervised/RL) | 3 |
| 004-013 | Learning rates (1e-5 to 1e-2) | 10 |
| 014-023 | Reward functions | 10 |
| 024-033 | Question types | 10 |
| 034-043 | Baseline configurations | 10 |
| 044-051 | Temperature (0.1 to 3.0) | 8 |
| 052-056 | Entropy coefficient (0.0 to 0.1) | 5 |
| 057-061 | Batch size (4 to 64) | 5 |
| **Total** | | **61** |

---

## STEP 3: Run All 61 Experiments 🚀

### Option A: Run ALL experiments (takes ~2-4 hours)
```powershell
python scripts/run_all_experiments.py --config_dir config/experiments/ --data_dir data/generated/ --output_dir experiments/results/
```

### Option B: Run specific range (faster)
```powershell
# Start from experiment 4, run max 10 experiments
python scripts/run_all_experiments.py --start_from 4 --max_experiments 10
```

### Option C: Run single experiment
```powershell
python scripts/run_experiment.py --config config/experiments/exp_003_rl_baseline.yaml
```

### Option D: Quick test with dummy models (seconds, for testing only)
```powershell
python scripts/run_all_experiments.py --use_dummy --max_experiments 5
```

---

## STEP 4: Increase Training Steps for Better Results

Current configs use only 50 steps (very fast, but low accuracy). To get meaningful results:

### Edit base_config.yaml to increase steps:
```yaml
# Change in config/base_config.yaml line 47:
max_steps: 500    # was 50, use 500-2000 for better results
```

### Or override from command line:
```powershell
python scripts/run_experiment.py --config config/experiments/exp_003_rl_baseline.yaml --max_steps 500
```

---

## STEP 5: Analyze Results 📊

### Generate analysis report:
```powershell
python scripts/analyze_results.py --results_dir experiments/results/ --output_dir experiments/analysis/ --plot
```

### This generates:
- `experiments/analysis/analysis_summary.json` - Full stats
- `experiments/analysis/results_table.md` - Markdown table
- `experiments/analysis/accuracy_comparison.png` - Bar chart
- `experiments/analysis/per_type_accuracy.png` - Per question type

---

## STEP 6: Generate Presentation Materials 📈

### Run presentation analysis:
```powershell
python scripts/analyze_presentation.py --results_dir experiments/results/ --output_dir experiments/analysis/
```

### Quick Accuracy Check:
```powershell
python check_accuracy.py
```

---

## STEP 7: Understanding the Results

### What is Ground Truth?
- **Ground truth** = The correct answer for each VQA sample
- Generated during data creation in `data/generated/*/metadata.json`
- Example: Image shows "red cube", Question "What color is the cube?", Answer = "red"

### What is the Benchmark?
1. **Frozen baseline** (exp_001): No training, just CLIP→T5, ~0% accuracy
2. **Supervised baseline** (exp_002): Cross-entropy on correct answers, ~15-40% with 500 steps
3. **RL (REINFORCE)** (exp_003-061): Policy gradient with reward shaping

### Why 61 Experiments are Meaningful:
| Category | Research Question |
|----------|-------------------|
| Method comparison (1-3) | Does RL outperform supervised learning? |
| Learning rate (4-13) | What's the optimal LR for RL? |
| Reward function (14-23) | Which reward signal works best? |
| Question types (24-33) | Which skills are easiest to compose? |
| Baselines (34-43) | Does variance reduction help? |
| Temperature (44-51) | Exploration vs exploitation tradeoff |
| Entropy (52-56) | Does entropy bonus help exploration? |
| Batch size (57-61) | Variance/speed tradeoff |

---

## STEP 8: Key Talking Points for Supervisor

### 1. Research Contribution
> "We extend the compositional skill learning paper by testing whether REINFORCE can compose frozen vision skills (CLIP) with trainable language skills (T5) without intermediate supervision."

### 2. Why CPU is Valid
> "By freezing CLIP and using T5-small, we eliminate the GPU bottleneck. The research question is about learning dynamics, not scale. Small-scale experiments reveal the same patterns."

### 3. Key Hypotheses Being Tested
- **H1**: RL outperforms supervised learning for compositional tasks
- **H2**: Progressive reward shaping accelerates learning  
- **H3**: Question type affects composability (spatial harder than color)
- **H4**: Variance reduction (baseline) is critical for RL stability

### 4. Expected Results Pattern
- Frozen: ~0% (no learning)
- Supervised: 15-40% (direct supervision)
- RL baseline: 10-30% (sparse reward is hard)
- RL + reward shaping: 20-45% (should beat sparse RL)
- RL + best hyperparams: Potentially beats supervised

### 5. Generalization Analysis
The project measures:
- **In-distribution**: Same color/shape combos as training
- **Generalization gap**: Performance on held-out attribute combinations

---

## QUICK COMMANDS CHEATSHEET

```powershell
# Setup verification
python -c "import torch; import transformers; print('OK')"

# Generate data
python scripts/prepare_data.py --num_train 5000 --num_val 1000 --num_test 1000

# Run single experiment
python scripts/run_experiment.py --config config/experiments/exp_003_rl_baseline.yaml

# Run all experiments
python scripts/run_all_experiments.py

# Run with more steps
python scripts/run_experiment.py --config config/experiments/exp_003_rl_baseline.yaml --max_steps 500

# Analyze results
python scripts/analyze_results.py --plot

# Quick accuracy check
python check_accuracy.py
```

---

## FINAL CHECKLIST BEFORE PRESENTATION

- [ ] All 61 experiments have run (check `experiments/results/`)
- [ ] Analysis plots generated (`experiments/analysis/*.png`)
- [ ] Results table created (`experiments/analysis/results_table.md`)
- [ ] Understand key findings:
  - [ ] Best method (RL vs Supervised)
  - [ ] Best learning rate
  - [ ] Best reward function
  - [ ] Hardest question type
- [ ] Prepare 2-3 slides:
  - Slide 1: Architecture diagram (README has ASCII art)
  - Slide 2: Accuracy comparison bar chart
  - Slide 3: Key findings table

---

## TROUBLESHOOTING

### "ModuleNotFoundError"
```powershell
pip install -r requirements.txt
```

### "CUDA not available" warning
This is expected! The project runs on CPU by design.

### Low accuracy (0%)
Increase `max_steps` to 500+ in config or command line.

### Experiments too slow
Reduce batch size to 8 or use `--use_dummy` for testing.

---

**Good luck with your presentation!** 🎓
