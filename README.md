# TARR: Training with Adaptive Randomised Response for Label-Private and Individually Fair Classification

Code for the CIKM '26 paper (see `CIKM26_paper_1001.pdf`).

TARR trains a classifier with two simultaneous guarantees:

- **Label differential privacy** (label-DP): training labels are protected by randomised response, so an adversary observing the trained model gains at most ε_p information about any individual label.
- **Bounded individual fairness** (IF) gap: predictions are approximately consistent when the sensitive feature of a test input changes, with the gap bounded via group privacy from the sensitive-feature perturbation budget ε_f.

The key mechanism is **Algorithm 1**: at each SGD step, both the sensitive feature column and the label are independently perturbed using randomised response with an *adaptive prior* that tracks the empirical distribution of previously sampled values.

---

## Repo layout

```
tarr/
├── tarr/
│   ├── rr.py       rr_with_prior + update_prior  (Algorithm 1 primitive)
│   ├── model.py    SimpleMLP  (6-layer FC, hidden widths [64,32,16,8,4])
│   ├── data.py     Dataset loaders for Adult, German Credit, Bank Marketing
│   └── eval.py     CNS (consistency score) and accuracy
├── train.py        tarr_train (Algorithm 1) and erm_train baseline
├── run_adult.py    Adult experiments  (sensitive: sex / race / age)
├── run_credit.py   German Credit experiments  (sensitive: age / sex)
├── run_bank.py     Bank Marketing experiments  (sensitive: age)
└── requirements.txt
```

---

## Install

```bash
pip install -r requirements.txt
```

Datasets are downloaded automatically on first use via `sklearn.datasets.fetch_openml` and cached under `~/scikit_learn_data/`.

---

## Running experiments

Each script accepts `--method tarr` (default) or `--method erm` and the relevant `--sensitive` attribute. Pass `--all` to run every combination for that dataset.

```bash
# Adult — three sensitive attributes
python run_adult.py --sensitive sex  --method tarr --eps_p 1.0 --eps_f 1.0
python run_adult.py --sensitive race --method tarr
python run_adult.py --sensitive age  --method tarr

# All Adult combinations (tarr + erm × sex/race/age)
python run_adult.py --all

# German Credit
python run_credit.py --sensitive age --method tarr
python run_credit.py --sensitive sex --method tarr
python run_credit.py --all

# Bank Marketing
python run_bank.py --method tarr
```

Key flags (shared across all run scripts):

| Flag | Default | Description |
|------|---------|-------------|
| `--eps_p` | 1.0 | Label-DP budget ε_p |
| `--eps_f` | 1.0 | Individual-fairness budget ε_f |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 64 | Mini-batch size |
| `--lr` | 0.01 | AdamW learning rate |
| `--seed` | 42 | Random seed |
| `--device` | auto | `cuda` or `cpu` |

---

## Metrics

**CNS** (Consistency Score, paper Equation 10): percentage of test inputs whose predicted label is unchanged when the binary sensitive feature is flipped. Higher is more individually fair. Reported in Table 1.

**ADF** (Adversarial Discrimination Finder, Zhang et al. [40]): success rate of a white-box adversarial search for discriminatory pairs. Requires the external ADF tool; not included here.

**FairQuant** (post-hoc verification, [26]): symbolic interval analysis to certify IF. Requires the external FairQuant tool; not included here.

**Accuracy**: standard test accuracy, reported in Table 3.

---

## Baselines

The paper compares against LFR [39], Rawlsian max-min [22], DP-SGD [1], LP-2ST [18], and LabelDP-Pro [19]. Those baselines are implemented in their respective original repositories. The ERM baseline (`--method erm`) is included here for reference.

---

## Algorithm sketch

```
prior_f ← uniform over {0, 1}           # sensitive-feature prior
prior_l ← uniform over {0, 1}           # label prior
for each training step t:
    (x_sen, x_non, y) ← next record
    x̌_sen ← RRWithPrior(x_sen, ε_f / t_max, prior_f)
    ỹ     ← RRWithPrior(y,     n·ε_p / t_max, prior_l)
    θ ← θ − γ · ∇_θ ℓ(model(x_non, x̌_sen), ỹ)
    prior_f ← prior_f − (prior_f − e_{x̌_sen}) / t
    prior_l ← prior_l − (prior_l − e_ỹ) / t
```

`RRWithPrior` samples category k with probability proportional to `prior[k] · exp(ε · 1[k == input])`, so the mechanism is ε-locally differentially private and shifts probability mass toward frequently observed categories as training progresses.
