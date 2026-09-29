# Deterministic MCQ Classification via Logit-Level Scoring (Flan-T5-XL)

A production-grade implementation for deterministic multiple-choice question (MCQ) answering using `google/flan-t5-xl`. Instead of allowing the model to produce free-form natural language and attempting regex parsing, this engine inspects raw token logits at generation step 0 to force an exact, discrete selection.

---

## 📌 Background & Motivation

Standard LLM text generation (`model.generate()`) presents notable hurdles when used in structured evaluation pipelines:
* **Output Variance:** Models may emit conversational filler (e.g., `"The answer is A"`, `"Option A."`, `"A: Blue"`), breaking downstream autograders or parsers.
* **Non-Determinism:** Sampling configurations can yield conflicting results across identical requests.
* **Latency Overhead:** Generating multi-token explanations is compute-intensive when only a classification decision is required.

By extracting the model's unnormalized logits directly over target vocabulary tokens (`A`, `B`, `C`, `D`), we achieve **O(1) decoding steps**, zero hallucination risk, and guaranteed deterministic outputs.

---

## 🛠️ How It Works

1. **Prompt Engineering:** The question and candidates are structured with strict label anchors:
   ```text
   Answer the following multiple choice question by giving only the option letter.

   Question: <q>
   A: <a>
   B: <b>
   C: <c>
   D: <d>

   Answer: