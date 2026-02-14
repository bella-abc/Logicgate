# LogicGate: Adaptive Rule-Based Modeling of Exogenous Effects for Time Series Forecasting

Official implementation of the paper **LogicGate: Adaptive Rule-Based Modeling of Exogenous Effects for Time Series Forecasting**.

## 1. Project Overview

LogicGate is a forecasting framework for exogenous-aware time series prediction.  
Compared with static rule sets and shallow fusion methods, this implementation focuses on:

- dynamic rule filtering based on current exogenous context,
- cross-modal fusion between time-series patches and rule semantics,
- dual-track self-optimization of rule quality during training.

![LogicGate Overview](./overview.png)

## 2. Installation

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python 3.10+
- PyTorch 2.0+

If you use `optimization` mode, set API key in environment variables:

```bash
# Linux/Mac
export OPENAI_API_KEY=your_key

# Windows PowerShell
$env:OPENAI_API_KEY="your_key"
```

## 3. Data and Rule Config

### Data

Put datasets under `dataset/` (or pass custom paths via `--root_path` and `--data_path`).

### Rule Config

- Default global rule file: `config/rule_patterns.json`
- Dataset-specific rule files: `config/rule_config/`
  - `rule_patterns_BE.json`
  - `rule_patterns_DE.json`
  - `rule_patterns_ETTh1.json`
  - `rule_patterns_FR.json`
  - `rule_patterns_NP.json`
  - `rule_patterns_wet_bulb.json`

Rule files are used by:

- `--rules_list` (load rule text for model input)
- `--rule_config_path` (rule-pattern filtering/analysis path)

## 4. Quick Start

### 4.1 Baseline Training

```bash
python train.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id demo_baseline \
  --model_comment baseline \
  --model logicgate \
  --data ETT_exog \
  --root_path ./dataset/electricity-price \
  --data_path NP.csv \
  --features MS \
  --target OT \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 3 --dec_in 3 --c_out 1 \
  --train_epochs 10 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --optimization_mode baseline
```

### 4.2 Monitoring Mode

```bash
python train.py ... --optimization_mode monitoring
```

### 4.3 Full Optimization Mode

```bash
python train.py ... \
  --optimization_mode optimization \
  --optimization_config config/optimization_config_stage4.json \
  --rules_list config/rule_patterns.json \
  --rule_config_path config/rule_patterns.json
```

Or use the stage-4 runner:

```bash
python run_optimization_stage4.py
```

### 4.4 Dataset Script Entry

```bash
python script/run_NP.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id np_demo \
  --model_comment test \
  --model logicgate \
  --data ETT_exog
```

## 5. Key Configurations

Important config files:

- `config/rule_patterns.json`
- `config/optimization_config.json`
- `config/optimization_config_stage4.json` (for stage-4 optimization flow)

Common arguments in `train.py`:

- `--optimization_mode`: `baseline | monitoring | optimization`
- `--rules_list`: rule file used for model rule text loading
- `--rule_config_path`: rule file path used in filtering/analysis
- `--optimization_config`: optimization pipeline config path
- `--rule_filter_method`: `basic | fast`
- `--num_exog_vars`: number of exogenous variables
