How the educational-video pipeline is evaluated: the benchmark dataset it's built and tested on, and the three feedback sources that judge its outputs.

---
## 1. Evaluation dataset (benchmark)

### Source

Grounded in China's national curriculum — the authoritative, single-source advantage over
prior work (which hand-waved difficulty levels). All content is anchored to:

- **《义务教育课程方案和课程标准（2022年版）》** — the Ministry of Education's compulsory- education curriculum standards (16 subject standards, 教材〔2022〕2号, in effect since fall 2022), and the matching high-school standards.
- Official textbooks (人教版 / 统编版 where applicable), available free from the
  **国家中小学智慧教育平台** (the Ministry-run platform) — the canonical, citable, legally
  clean source for the actual textbook PDFs.

**Reproducibility note:** the textbooks are mid-transition (2024–2026 rollout under the
2022 标准 revision), so the exact edition/year used must be pinned, or the benchmark won't be reproducible.

### Subject scope (objective-knowledge subjects)

Subjects were screened for *objectivity* — whether a concept has a defensible right answer the LLM-judge can score — and for *content sensitivity*. The 2022 standards are explicitly built on 立德树人 and Marxist guidance, with value-cultivation a mandated function of several subjects; those are excluded.

**Elementary (小学):**

- 语文 (Chinese)
- 数学 (Math)
- 英语 (English)
- 科学 (Science)

**Middle (初中):**

- 语文 (Chinese)
- 数学 (Math)
- 英语 (English)
- 物理 (Physics)
- 化学 (Chemistry)
- 生物 (Biology)
- 历史 (History) ⚠️
- 地理 (Geography) ⚠️

**High (高中):**

- 语文 (Chinese)
- 数学 (Math)
- 英语 (English)
- 物理 (Physics)
- 化学 (Chemistry)
- 生物 (Biology)
- 历史 (History) ⚠️
- 地理 (Geography) ⚠️
- 信息技术 (Information Technology)

⚠️ **History & Geography** — interpretive/value-laden content (no clean ground truth for the LLM-judge; lean on human evaluation) and politically sensitive material per the 2022 standards, which Western-model pipelines may handle inconsistently.

---
## 2. Feedback sources

Three layers, increasing in strength and cost.

### a. LLM-judge (automated, scales)

An LLM/VLM scores videos across evaluation dimensions. Runs cheaply on the full benchmark, produces the main quantitative table, and gives an apples-to-apples comparison to prior baselines. Uses a current frontier model as judge (a deliberate upgrade over prior work's weaker judge); rubric is adapted per content type. Runs continuously *during* development.

### b. Real students (various grades)

Learners across the benchmark's grade bands. Strong form is a **learning-outcome** study (pre-test → watch → post-test → measure gain), we can do a quality rating, but then also real studies on top of that, this needs recruitment across age bands and
consent handling for minors.

### c. Real teachers (various grades)

Teachers across grade bands judge **usefulness / adoption** — "would I actually use this in my class?" Aligns with the pipeline's human-in-the-loop design (teachers are the users).

**Effort note:** 
- (a) runs unattended and solo; 
- (b) and (c) are the logistically heavy human studies (recruitment, consent, scheduling, test design) where collaborators / an advising lab earn their place.