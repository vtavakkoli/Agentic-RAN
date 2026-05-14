# BioSwarm Urban Monitoring

BioSwarm Urban Monitoring is a Docker-based benchmark for **uncertainty-aware multi-agent urban risk monitoring**. It simulates urban risk zones and evaluates classical agents, bio-inspired swarm agents, SOTA-style multi-agent baselines, and the proposed **UA-HBAS-PPO** policy under realistic monitoring conditions.

The benchmark covers noisy sensing, communication dropout, partial observability, dynamic risk drift, multiple city-grid sizes, different team sizes, and battery/step-limited agents.

---

## Main Objective

The purpose of this repository is to evaluate whether an **uncertainty-aware hybrid bee-ant swarm policy with PPO learning** can improve urban risk monitoring compared with classical and learning-based baselines.

The complete workflow is:

```text
Generate dataset
→ Train UA-HBAS-PPO and ablations
→ Evaluate all agents
→ Build HTML reports
```

---

## Proposed Method

The main proposed method is:

```text
UA-HBAS-PPO
```

which means:

```text
Uncertainty-Aware Hybrid Bee-Ant Swarm with PPO Policy Learning
```

The method combines:

- bee-inspired local exploitation,
- ant-inspired exploration and pheromone-style guidance,
- uncertainty-aware zone prioritization,
- PPO-based policy learning,
- imitation pretraining from teacher agents,
- behavior-diversity regularization.

---

## Important Pipeline Conditions

### Condition 1: Dataset generation must run first

Training and testing require the generated scenario manifest:

```text
dataset/scenario_manifest.csv
```

Create it with:

```bash
docker-compose up --build generate-dataset
```

or:

```bash
docker compose up --build generate-dataset
```

---

### Condition 2: Training must run before strict testing

The test configuration uses strict checkpoint validation:

```yaml
strict_checkpoint_validation: true
```

Therefore, the following files must exist before testing:

```text
results/train/checkpoints/ua_hbas_ppo_full.pt
results/train/checkpoints/ua_hbas_ppo_no_imitation.pt
results/train/checkpoints/ua_hbas_ppo_no_diversity.pt
```

Create them with:

```bash
docker-compose up --build train
```

---

### Condition 3: Use the same generated dataset for all models

Do not regenerate the dataset between model evaluations. A fair comparison requires:

```text
same scenario_manifest.csv
same split
same seeds
same grid sizes
same number of agents
same uncertainty levels
same test scenarios
```

---

### Condition 4: `run-all` is the recommended full pipeline

The following command executes the complete reproducible workflow:

```bash
docker-compose up --build run-all
```

It runs:

```text
1. Dataset generation
2. Dataset report creation
3. PPO training
4. Training report creation
5. Agent evaluation
6. Final test report creation
```

---

### Condition 5: Publication claims must be conservative

Recommended wording:

```text
UA-HBAS-PPO achieves the best operational trade-off among the evaluated baselines under the tested uncertainty settings.
```

Avoid unsupported claims such as:

```text
UA-HBAS-PPO is better than all existing SOTA methods.
```

unless all SOTA-style baselines run successfully and statistical validation supports that claim.

---

## Repository Structure

```text
BioSwarm-Urban-Monitoring-main/
│
├── configs/
│   ├── dataset.yaml
│   ├── ppo_train.yaml
│   └── test.yaml
│
├── dataset/
│   ├── .gitkeep
│   └── scenario_manifest.csv              # generated
│
├── docker/
│   └── Dockerfile
│
├── experiments/
│   ├── generate_dataset.py
│   ├── train_ppo.py
│   └── evaluate_agents.py
│
├── reports/
│   ├── build_dataset_report.py
│   ├── build_train_report.py
│   └── build_report.py
│
├── results/
│   ├── generate_dataset/
│   ├── train/
│   └── test/
│
├── src/
│   ├── agents/
│   ├── environment/
│   ├── evaluation/
│   └── utils/
│
├── docker-compose.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Docker Compose Commands

Use either the old syntax:

```bash
docker-compose ...
```

or the newer syntax:

```bash
docker compose ...
```

Both are shown below.

---

### 1. Generate Dataset

```bash
docker-compose up --build generate-dataset
```

or:

```bash
docker compose up --build generate-dataset
```

This service runs:

```bash
python experiments/generate_dataset.py --config configs/dataset.yaml
python reports/build_dataset_report.py --root results/generate_dataset
```

Expected outputs:

```text
dataset/scenario_manifest.csv
results/generate_dataset/
results/generate_dataset/report.html
```

---

### 2. Train UA-HBAS-PPO

```bash
docker-compose up --build train
```

or:

```bash
docker compose up --build train
```

This service runs:

```bash
python experiments/train_ppo.py --config configs/ppo_train.yaml
python reports/build_train_report.py --root results/train
```

Expected outputs:

```text
results/train/checkpoints/
results/train/report.html
```

Important checkpoint files:

```text
results/train/checkpoints/ua_hbas_ppo_full.pt
results/train/checkpoints/ua_hbas_ppo_no_imitation.pt
results/train/checkpoints/ua_hbas_ppo_no_diversity.pt
```

---

### 3. Test / Evaluate Agents

```bash
docker-compose up --build test
```

or:

```bash
docker compose up --build test
```

This service runs:

```bash
python experiments/evaluate_agents.py --config configs/test.yaml
python reports/build_report.py --root results/test --output results/test/report.html
```

Expected outputs:

```text
results/test/
results/test/report.html
```

---

### 4. Run Complete Pipeline

```bash
docker-compose up --build run-all
```

or:

```bash
docker compose up --build run-all
```

This is the recommended command for a complete experiment.

It performs:

```text
generate-dataset
→ dataset report
→ train
→ training report
→ test
→ final report
```

---

### 5. Stop Containers

```bash
docker-compose down
```

or:

```bash
docker compose down
```

---

### 6. Rebuild Without Cache

Use this after changing dependencies, the Dockerfile, or package versions:

```bash
docker-compose build --no-cache
```

or:

```bash
docker compose build --no-cache
```

Then run the required service again, for example:

```bash
docker-compose up generate-dataset
```

---

### 7. Run One Service Without Rebuilding

```bash
docker-compose up generate-dataset
```

```bash
docker-compose up train
```

```bash
docker-compose up test
```

---

### 8. Clean Full Reproducible Run

Linux/macOS:

```bash
docker-compose down
rm -rf results/generate_dataset results/train results/test dataset/scenario_manifest.csv
docker-compose up --build run-all
```

Windows PowerShell:

```powershell
docker-compose down
Remove-Item -Recurse -Force results\generate_dataset, results\train, results\test -ErrorAction SilentlyContinue
Remove-Item -Force dataset\scenario_manifest.csv -ErrorAction SilentlyContinue
docker-compose up --build run-all
```

---

## Local Python Commands

Docker is recommended, but the pipeline can also run locally.

### Create Environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Generate Dataset Locally

```bash
python experiments/generate_dataset.py --config configs/dataset.yaml
python reports/build_dataset_report.py --root results/generate_dataset
```

---

### Train Locally

```bash
python experiments/train_ppo.py --config configs/ppo_train.yaml
python reports/build_train_report.py --root results/train
```

---

### Test Locally

```bash
python experiments/evaluate_agents.py --config configs/test.yaml
python reports/build_report.py --root results/test --output results/test/report.html
```

---

### Full Local Pipeline

Linux/macOS:

```bash
python experiments/generate_dataset.py --config configs/dataset.yaml && \
python reports/build_dataset_report.py --root results/generate_dataset && \
python experiments/train_ppo.py --config configs/ppo_train.yaml && \
python reports/build_train_report.py --root results/train && \
python experiments/evaluate_agents.py --config configs/test.yaml && \
python reports/build_report.py --root results/test --output results/test/report.html
```

Windows PowerShell:

```powershell
python experiments/generate_dataset.py --config configs/dataset.yaml; `
python reports/build_dataset_report.py --root results/generate_dataset; `
python experiments/train_ppo.py --config configs/ppo_train.yaml; `
python reports/build_train_report.py --root results/train; `
python experiments/evaluate_agents.py --config configs/test.yaml; `
python reports/build_report.py --root results/test --output results/test/report.html
```

---

## Configuration Files

### `configs/dataset.yaml`

Controls dataset generation.

Important parameters:

```yaml
seed: 42
risk_zones: 5
max_steps: 200
high_risk_threshold: 0.75
```

Dataset splits:

```text
train
validation
test
```

Each split controls:

```text
seeds
grid_sizes
n_agents_list
sensor_noise_levels
communication_dropout_levels
partial_observability_levels
risk_drift_levels
```

---

### `configs/ppo_train.yaml`

Controls PPO training and ablations.

Important parameters:

```yaml
dataset_split: train
validation_split: validation
episodes: 180
use_imitation_pretraining: true
use_behavior_diversity_regularization: true
train_ablation_checkpoints: true
```

Teacher agents:

```yaml
teacher_agents:
  - AntSwarmAgent
  - GraphMARLAgent
```

Training output:

```text
results/train/
```

---

### `configs/test.yaml`

Controls final evaluation.

Important parameters:

```yaml
dataset_split: test
output_root: results/test
strict_checkpoint_validation: true
quick_scenarios: 3
```

Checkpoint paths:

```yaml
ppo_checkpoint: results/train/checkpoints/ua_hbas_ppo_full.pt
ppo_checkpoint_no_imitation: results/train/checkpoints/ua_hbas_ppo_no_imitation.pt
ppo_checkpoint_no_diversity: results/train/checkpoints/ua_hbas_ppo_no_diversity.pt
```

---

## Agentic Policy Section

The project evaluates a policy layer that chooses monitoring actions based on uncertain observations from the city grid.

```text
Urban risk state
→ noisy / partial observations
→ agentic policy
→ movement and monitoring actions
→ coverage, detection, energy, and communication metrics
→ policy ranking
```

---

## Agents Included

### Classical Baselines

```text
RandomAgent
GreedyAgent
CoverageAgent
```

### Bio-Inspired Baselines

```text
AntSwarmAgent
BeeSwarmAgent
PSOAgent
UncertaintyAwareBeeAntSwarmAgent
```

### Learning-Based and SOTA-Style Baselines

```text
GraphMARLAgent
PPOAgent
IPPOAgent
MAPPOAgent
QMIXAgent
WQMIXAgent
MADDPGAgent
HAPPOAgent
MATAgent
GRPO
GRPO-MAT-HBAS
```

### Proposed Methods and Ablations

```text
UA-HBAS-NoPPO
UA-HBAS-PPO
UA-HBAS-w/o-Imitation
UA-HBAS-w/o-Diversity
```

Deprecated methods such as Bio-GRPO and ZA-Bio-GRPO are excluded from the official publication benchmark because they are duplicated, unstable, or not sufficiently distinct from the retained GRPO method.

---

## Metrics

The benchmark reports operational monitoring metrics such as:

```text
detection_rate
high_risk_detection_rate
coverage_ratio
average_time_to_first_detection
missed_risk_zones
energy_consumption
redundant_coverage
communication_efficiency
composite_score
runtime_seconds
```

For publication-style comparison, prioritize:

```text
composite_score
high_risk_detection_rate
coverage_ratio
missed_risk_zones
energy_consumption
communication_efficiency
runtime_seconds
```

---

## Expected Outputs

After running the complete pipeline, the expected output structure is:

```text
dataset/scenario_manifest.csv

results/generate_dataset/
results/generate_dataset/report.html

results/train/
results/train/checkpoints/
results/train/report.html

results/test/
results/test/report.html
```

The main final report is:

```text
results/test/report.html
```

---

## Troubleshooting

### Dataset manifest is missing

Problem:

```text
dataset/scenario_manifest.csv not found
```

Fix:

```bash
docker-compose up --build generate-dataset
```

---

### PPO checkpoint is missing

Problem:

```text
results/train/checkpoints/ua_hbas_ppo_full.pt not found
```

Fix:

```bash
docker-compose up --build train
```

or run everything:

```bash
docker-compose up --build run-all
```

---

### Results are not comparable

This usually happens if the dataset was regenerated between evaluations.

Fix:

```bash
docker-compose down
rm -rf results/test
docker-compose up --build test
```

Do not regenerate `dataset/scenario_manifest.csv` unless you want to restart the full benchmark.

---

### Docker Compose command does not work

Try the newer Docker syntax:

```bash
docker compose up --build run-all
```

instead of:

```bash
docker-compose up --build run-all
```

---

## Recommended Workflow

For daily development:

```bash
docker-compose up --build generate-dataset
docker-compose up --build train
docker-compose up --build test
```

For a clean final benchmark:

```bash
docker-compose down
rm -rf results/generate_dataset results/train results/test dataset/scenario_manifest.csv
docker-compose up --build run-all
```

Then open:

```text
results/test/report.html
```

---

## License

This project uses the license provided in:

```text
LICENSE
```

---

## Author

Written by:

```text
Dr. techn. Vahid Tavakkoli
Dr. techn. Kabeh Mohsenzadegan
Dr. techn. Kabeh Mohsenzadegan
```
