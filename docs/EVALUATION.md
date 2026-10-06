# Evaluation

How rag-eval-lab is measured, the experiments run so far, and what the results do and don't show. All numbers below come from real runs saved in `results/` (not committed). Unless stated: `all-MiniLM-L6-v2` embeddings, k=4, and Claude Sonnet 5.5 as the judge. Experiments 1 to 5 also used Sonnet 5.5 as the generator; experiment 6 compares it with Claude Haiku 4.5.

## What gets measured

| Metric | Question it answers | How it's scored |
|---|---|---|
| `retrieval_hit_rate` | Did the chunk containing the answer come back in the top k? | Every `evidence` phrase must appear in a retrieved chunk from the expected `source` file |
| `mrr` | How high was that chunk ranked? | Mean of 1/rank of the first chunk containing the first evidence phrase (1.0 = always first) |
| `correct_refusal_rate` | Did it decline questions the documents can't answer? | Judge classifies the answer as full, partial, or declined |
| `false_refusal_rate` | Did it decline questions it *could* answer? | Same classification, on answerable questions; lower is better |
| `correctness` | Does the answer match the reference? | LLM judge |
| `faithfulness` | Is every claim supported by the retrieved passages? | LLM judge |

Retrieval metrics are deterministic. The others come from a judge: a second LLM call, pinned to its own model (`JUDGE_MODEL`), that returns a JSON verdict with an answer type, correctness, faithfulness, and a one-line reason. Pinning the judge means changing the generator model never silently changes the grader too.

A decline on an answerable question is scored as incorrect. With `--no-judge`, refusals fall back to a phrase check (the answer starts with "I don't know"), which is cheaper but cruder; see finding 8. Experiments 1 to 5 used that phrase check.

## The golden sets

- **`golden.jsonl` (15 questions):** direct factual questions, plus 3 unanswerable ones.
- **`golden_hard.jsonl` (12 questions):** built to break things:
  - paraphrases that avoid the documents' wording ("vacation" instead of "paid time off")
  - multi-hop questions needing two chunks, or arithmetic across facts
  - a number trap ("Can I carry over 10 days?" when the limit is 5)
  - a false premise ("Since MFA is optional...")
  - near-miss unanswerables close to real topics (office hours vs. support hours)

## Experiments

### 1. Baseline: one document

With only the sample handbook indexed, both sets scored 1.0 on every metric.

**Interpretation:** the test was too easy, not the system perfect. The handbook is about 8 chunks, so k=4 retrieves half the corpus on every question and a retrieval miss is nearly impossible. The hard questions also scored 1.0, which showed generation was strong but said nothing about retrieval.

### 2. Shrinking k

| Hard set | k=1 | k=2 |
|---|---|---|
| retrieval_hit_rate | 0.889 | 1.0 |

At k=1 only the multi-hop question (h05, whose answer spans two chunks) failed retrieval. Its answer was still the ideal behavior: it gave the half it could support and said it didn't know the other half.

**Finding:** k must be at least the number of chunks an answer needs. On this corpus k=2 is the minimum; higher k only adds tokens and noise to every prompt.

### 3. Distractor documents

Four documents were added to give retrieval real competition, each designed to collide with specific questions: another company's HR handbook (different PTO, a part-time policy, parental leave, pet insurance), a competitor's robot spec sheet (different load and battery figures), a generic security template that makes MFA optional, and a generic robot buying guide.

First results (k=4, before fixes):

| | Easy set | Hard set |
|---|---|---|
| faithfulness | 0.733 | 0.583 |
| correctness | 1.0 | 0.889 |
| mrr | 0.917 | 0.833 |
| correct_refusal_rate | 1.0 | 0.667 |

Most of this drop turned out to be the evaluation, not the system (see findings 1 and 3 below).

### 4. Fixing the judge

After giving the judge the same source labels the generator sees, on the easy set:

| | Before | After |
|---|---|---|
| faithfulness | 0.733 | 1.0 |
| mrr | 0.917 | 0.917 |

Retrieval scores didn't move, as expected: the fix changed grading, not the system. The remaining MRR gap is real: distractor chunks outrank the correct one on several questions.

### 5. Scope decision and an attribution rule

The app is a personal, multi-document assistant: users upload whatever they want, so it can't assume which organization a question is about. Under that scope, "Acme allows 5 days, Northwind allows 10" is the correct answer to an unscoped question. References for ambiguous questions were updated to expect per-source attribution, and the judge rubric was told that correctly attributed facts about another source are not contradictions.

Then a rule was added to the generation prompt: never merge facts across documents; answer separately per source when the question doesn't say which one it means. The two changes were tested separately so their effects wouldn't be confused.

Hard set, average of 3 runs each:

| | Pass 1: new labels, old prompt | Pass 2: + attribution rule |
|---|---|---|
| correctness | 1.0 | 1.0 |
| faithfulness | 0.972 | 1.0 |
| correct_refusal_rate | 1.0 | 1.0 |
| mrr | 0.80 | 0.80 |

The difference is one unsupported claim in 36 answers (Pass 1, run 1: an added remark that totals "exclude rolled-over PTO," which no document says) versus zero. The rule is kept as good practice, but one answer is not enough evidence to call it a measured improvement.

MRR reads 0.80 rather than 0.833 because the part-time PTO question became answerable from the other company's handbook, and its chunk ranks second.

### 6. Model comparison: Sonnet 5.5 vs. Haiku 4.5

The generator was switched to Claude Haiku 4.5 with the judge held at Sonnet 5.5, so only one thing changed.

**Haiku on the original prompt** declined two answerable questions in all 3 runs, in a "refuse, then explain" pattern: it opened with the refusal phrase and then described what the passages did say. On h07 (the false-premise MFA question) it saw the conflicting security template and backed off instead of correcting the premise.

**A prompt fix:** the refusal rule was rewritten to refuse only when the passages contain nothing that answers the question as asked, and otherwise to answer what is supported, state what is missing, and correct false premises.

| Haiku 4.5, phrase-based refusal check | Old prompt | New prompt |
|---|---|---|
| false_refusal_rate | 0.2 | 0.1 |
| correctness | 0.8 | 0.867 |
| correct_refusal_rate | 1.0 | 1.0 |
| faithfulness | 1.0 | 0.944 |

h07 went from declined in 3 of 3 runs to answered correctly in 3 of 3. The counter-metric held: near-miss questions were still refused. But faithfulness dipped. Once Haiku answered h07, it sometimes added that the security template "is not Acme's policy." That is true of the full document, but the sentence saying so sits in a chunk that was not retrieved; Haiku inferred it from the filename. Pushing a model to answer more trades a little faithfulness risk for fewer false refusals.

**Regression check on Sonnet** with the new prompt showed `correct_refusal_rate` falling to 0.5. Reading the answers showed the opposite of a regression: on the office-hours trap Sonnet wrote "the passages don't state when the Austin office opens," then noted the support hours as support hours. A correct decline, phrased without the exact refusal string. That led to replacing the phrase check with a judge classification (finding 8).

**Final comparison**, new prompt, behavior-based refusal metric, 3 runs each:

| | Haiku 4.5 | Sonnet 5.5 |
|---|---|---|
| correctness | 0.90 | 1.0 |
| faithfulness | 0.972 | 1.0 |
| correct_refusal_rate | 1.0 | 1.0 |
| false_refusal_rate | 0.10 | 0.0 |
| avg latency per question | ~1.9 s | ~4.0 s |

Haiku's remaining miss is h08 ("Who gets to see customer information?"), declined in all 3 runs: the documents answer it partly (access is least-privilege) and Haiku didn't say so, while Sonnet did every time. Its one faithfulness miss is the h07 filename inference above, in 1 of 3 runs.

**Decision:** Sonnet 5.5 is the default generator. Haiku 4.5 is a supported option for lower-stakes questions where speed and cost matter more than handling partial or conflicting evidence. Both send the same data to the same API, so the choice is about answer quality, not privacy.

The prompt was not tuned further for h08. Rewriting a prompt until every golden question passes fits it to these 12 questions rather than to real users; the better response is more questions of that type.

## Findings

1. **The judge must see exactly what the generator saw.** The generator received chunks labeled with their source file; the judge received bare text. It then graded every correct "Acme says..." attribution as unsupported. This was invisible with one document and dominated the scores with five.

2. **Refusal detection needs care.** The first version counted any answer containing "I don't know" as a refusal, so a good partial answer registered as a false refusal (0.111 at k=1). Refusals now require the phrase at the start of the answer; partial answers get their own flag.

3. **Golden labels go stale when the corpus changes.** Questions written for a one-document corpus ("unanswerable: part-time PTO") became answerable once another company's handbook arrived. The model's answers were right; the labels were wrong. Reviewing the golden set is part of adding documents.

4. **Single runs are noisy.** The same question flipped between pass and fail across identical configurations, from variation in both the answers and the judge. Since Anthropic SDK 1.0 removed `temperature`, it can't be pinned. With 12 questions one flip is about 8 points, so every comparison above uses repeated runs.

5. **Naming the subject scopes the answer.** Every question that named the company (pet insurance, parental leave, office hours) was refused correctly even when a distractor document covered the topic for another company. Ambiguity, not distractors as such, caused the attribution answers.

6. **A distance cutoff is a coarse filter.** With `--max-distance 1.2`, two unanswerable questions were refused in about 0.1 seconds without an LLM call. A third (pet insurance) was topically close to the PTO chunk, passed the cutoff, and was refused by the model instead. The model's refusal is the real safeguard; the cutoff saves cost on clearly off-topic questions.

7. **A metric can hide the failure it should count.** Correctness was first averaged only over judged answers, and refusals skipped the judge, so a false refusal simply dropped out and Haiku showed a perfect 1.0. Counting a decline on an answerable question as incorrect gives its true 0.8 on the original prompt.

8. **Score behavior, not wording.** Detecting refusals by matching "I don't know" misread both models in opposite directions: Haiku opened with the phrase and then partly answered (counted as a refusal), while Sonnet declined without the phrase (counted as an answer). Forcing exact phrasing through the prompt would tune the model to the metric. Instead the judge classifies each answer as full, partial, or declined.

9. **Record the configuration in every report.** The first runs of the day were meant to use Haiku but actually ran on Sonnet: a config setting was read under a different name than the code expected, so the code fell back to its built-in default. Reports now record the generator and judge models, which would have exposed this immediately.

## Validating the judge

An LLM judge is only as trustworthy as its verdicts, and here it is the same model family as the generator, which risks favoring its own style and sharing its blind spots. Guards in place:

- **Human spot-check.** Answers for the arithmetic question, the false-premise question, and the office-hours trap were read by hand; all three verdicts were correct.
- **Visible reasoning.** Every verdict includes a one-line reason, which is how the source-label bug was found.

- **A separate judge model.** The judge is pinned to Sonnet 5.5 regardless of which model generates answers, so Haiku's answers were graded by a different, stronger model.

Planned: negative controls (deliberately wrong answers that the judge must mark false) and a judge from a different provider.

## Limitations

- 27 questions total is small. Treat differences under about 10 points between single runs as noise.
- When the generator is also Sonnet 5.5, the judge grades its own model's answers.
- The corpus is synthetic and small; real document collections will make retrieval harder and MRR more informative.
- Correctness and faithfulness are binary. Claim-level scoring (as in Ragas) would distinguish one unsupported claim from many.
- Latency figures are dominated by API time and vary run to run; the first question of each run includes model loading and is several seconds slower. Use the median.

## Reproducing

```bash
python -m rag_eval.ingest
for i in 1 2 3; do python -m rag_eval.evaluate --golden data/eval/golden_hard.jsonl; done

# the same with Haiku as the generator (judge stays Sonnet)
for i in 1 2 3; do LLM_MODEL=claude-haiku-4-5-20251001 python -m rag_eval.evaluate --golden data/eval/golden_hard.jsonl; done
```

Each run writes a JSON report to `results/`. Rows use Ragas field names (`user_input`, `response`, `retrieved_contexts`, `reference`) so the same runs can be scored with Ragas.
