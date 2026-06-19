# VeraBench Dataset Card

## Summary

VeraBench is a compact Chinese benchmark for verifiable retrieval-augmented
reasoning. It is designed to test behavior that conventional answer-only RAG
benchmarks often undermeasure: evidence attribution, multi-document synthesis,
temporal reasoning, abstention, evidence conflict handling, and correction of
misleading premises.

VeraBench v1.1.2 contains:

| Item | Count |
| --- | ---: |
| Corpus documents | 57 |
| Questions | 152 |
| Ground-truth claims | 328 |
| Evidence references | 208 |
| Multi-hop questions | 51 |
| Questions with annotated evidence conflicts | 13 |
| Gold conflict pairs | 15 |
| Evidence references traceable to corpus | 208 / 208 |
| Shared-evidence dependency components | 27 |

The corpus dates range from 2019-10-23 to 2025-03-10. It covers technology,
AI policy, semiconductors, climate and energy, quantum computing, biomedicine,
and space exploration.

## Version

- Dataset version: `1.1.2`
- Corpus SHA-256:
  `bb99ce4d8b4ba7ee5a938595c7c786a8ad33601e4f096bec1bed844c43b6a8f3`
- Questions SHA-256:
  `c19e7401cbcd2526fb1e085c911d61095c8ce5bd19a664c13698a08024436882`

Run `verarag-validate-benchmark` before publishing results. The validator checks
schema constraints, ontology alignment, evidence references, conflict pairs,
unique IDs, fingerprints, and synchronization between repository and packaged
copies. It also verifies that each evidence span is a continuous corpus
substring or an ordered sequence of corpus substrings separated by ellipses.
Run `verarag-audit-contamination --reference <local-corpus>` when local model
training, prompt-development, or public dump corpora are available; it reports
exact and near-duplicate benchmark text overlap against those supplied files,
including reference excerpts and exact-match character offsets for reviewer
inspection.

## Question Distribution

| Type | Count | Expected capability |
| --- | ---: | --- |
| `single_evidence` | 27 | Retrieve and cite one factual source |
| `multi_evidence` | 26 | Synthesize multiple evidence items |
| `conflict` | 11 | Identify and explain incompatible claims |
| `temporal` | 25 | Respect dates, updates, and historical state |
| `unanswerable` | 26 | Abstain when the corpus lacks the answer |
| `misleading` | 37 | Reject or narrow a false user premise |

Difficulty distribution is 59 easy, 77 medium, and 16 hard questions. Expected
behavior distribution is 78 answer-with-citation, 37 premise correction, 26
abstention, and 11 answer-with-conflict-note questions.

## Data Format

`corpus.jsonl` documents contain:

- `doc_id`, `title`, `content`;
- scenario-level `source`, `date`, `author`, and optional `url`;
- `entities` and `tags`.

`questions.jsonl` records contain:

- question type, text, difficulty, and multi-hop flag;
- a reference answer and claim-level support annotations;
- evidence spans and categories;
- expected evidence-conflict pairs;
- expected answer behavior and tags;
- migration rationale where v1.1 changed ontology or difficulty, plus
  wording-clarification rationale where v1.1.1 repaired an ambiguous question.

Evidence categories comprise 166 supporting, 11 conflicting, 10 outdated, and
21 partial references. The 328 ground-truth claims comprise 326 supported and
2 refuted claims.

## Construction

The benchmark is manually curated as a controlled evaluation corpus. Passages
are concise benchmark summaries rather than guaranteed verbatim copies of the
linked upstream pages. Some documents and claims are intentionally outdated,
low-reliability, fictional, or misleading so that systems must reason about
provenance and disagreement instead of treating every retrieved statement as
equally true. For example, the StarTech scenario is synthetic.

The `source`, `author`, and `url` fields are provenance or scenario metadata;
they are not an assertion that every passage is an exact publication excerpt.
Users should consult authoritative upstream sources before relying on any
benchmark statement as current real-world fact.

Version 1.1 reviewed all question types and separates:

1. evidence-evidence conflict, where two represented claims are incompatible;
2. premise refutation, where evidence agrees but rejects an inference embedded
   in the user's question.

This distinction prevents misleading questions from artificially inflating
conflict-detection scores.

Version 1.1.1 is a wording-only benchmark patch. It clarifies V084 so the
question asks for 2023 renewable-energy additions as reported by the early-2024
source, matching its gold evidence and answer. It does not change corpus size,
question count, question distribution, or metric implementations.

Version 1.1.2 is an annotation-traceability patch. It repairs 25 evidence spans
whose abbreviated text omitted qualifiers or listed excerpts out of document
order. It does not change questions, answers, evidence IDs, conflict labels,
behavior labels, or metric implementations. After migration, 129 evidence
references are continuous exact substrings and 79 are ordered multi-segment
references; zero are untraceable.

## Intended Uses

VeraBench is intended for:

- regression testing of Chinese RAG pipelines;
- research on evidence-grounded generation and abstention;
- conflict detector and verifier diagnostics;
- behavior and confidence-calibration analysis;
- controlled ablations of retrieval, verification, and reasoning components.

It is not intended to:

- establish broad Chinese language understanding or factuality;
- replace evaluation on large public benchmarks;
- serve as a current factual knowledge base;
- support medical, legal, financial, or policy decisions;
- provide sufficient training data for a general-purpose detector.

## Evaluation

Reports should include at least:

- benchmark version and both fingerprints;
- model/provider, configuration hash, and implementation hash;
- answer metric version;
- answer F1, evidence recall/precision, behavior accuracy;
- conflict TP/FP/FN and premise-refutation TP/FP/FN;
- ECE/Brier calibration metrics and error count.
- deterministic bootstrap confidence intervals, including method, resample
  count, confidence level, and seed;
- shared-evidence dependency-cluster sensitivity intervals;
- paired question-level inference when claiming improvement over another run.

VeraBench v1.1.2 uses `soft-f1-v2`. Results from v1.0, v1.1, or another metric
version must be labeled non-comparable unless explicitly analyzed as a
historical report. Partitioned reports should be combined with
`verarag-merge-reports --require-complete`, which validates provenance and
question coverage before recomputing aggregate metrics.

`verarag-leaderboard` defaults to publication-oriented integrity checks. It
rejects demo identity reports, partial or errored runs, reports missing current
reproducibility metadata, and mixed benchmark fingerprints or metric versions.
The `--allow-demo`, `--allow-incomplete`, `--allow-unverified`, and
`--allow-mixed-benchmarks` flags are escape hatches for explicitly labeled
diagnostic or historical tables, not normal leaderboard publication.

New reports use `stratified-question-bootstrap-v1`: 2,000 percentile-bootstrap
resamples stratified by question type, with 95% confidence and seed 1729.
They also report `evidence-cluster-bootstrap-v1`, which resamples the 27
connected components induced by shared gold document IDs. This sensitivity
analysis addresses within-corpus dependence; it does not make the benchmark
representative of a broader task population.
System-to-system claims should use `verarag-compare-reports`, which performs
paired stratified and paired evidence-cluster resampling over identical
question IDs. These intervals describe uncertainty within this benchmark
sample and must not be presented as proof of performance on the broader
Chinese RAG population.

## Known Limitations

- The benchmark is small and topic-skewed; confidence intervals can be wide.
- The 152 questions form only 27 connected components under shared gold
  documents; question-level intervals alone overstate the number of independent
  evidence contexts. Dependency-cluster intervals should be reported too.
- It is Chinese-only and does not measure cross-lingual behavior.
- The corpus is controlled and much shorter than production knowledge bases.
- The source mix is scenario-oriented: 28 report, 14 paper, 6 news, 5 official,
  2 blog, and 2 wiki labels. Thirty documents have no URL field.
- Some passages are synthetic or intentionally adversarial.
- No independent inter-annotator agreement study has been published.
- A local contamination-audit tool is provided, but no exhaustive audit against
  proprietary or unknown model-training corpora has been completed.
- Difficulty labels are rubric-based and have not yet been validated against
  human or multi-model response statistics.
- Temporal statements age; the fixed fingerprints preserve reproducibility,
  not present-day factual freshness.
- `soft-f1-v2` is a heuristic lexical metric and should be interpreted with
  behavior and evidence metrics, not in isolation.
- Repeated development against the test set can overfit project-specific rules.
  Future releases should add a hidden or independently maintained test split.

## Licensing and Citation

The benchmark files are distributed with the repository under its MIT license.
Source names and URLs are included for attribution and scenario provenance;
upstream materials remain subject to their own terms. Cite the repository using
`CITATION.cff`, and report the dataset version and fingerprints with all
published results.

## Maintenance

Changes that alter annotations, corpus text, question text, or scoring semantics
require a benchmark version update, regenerated fingerprints, synchronized
repository/package data, migration notes, and a compatibility statement in the
results documentation.
