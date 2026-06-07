# The Unofficial Guide — Project 1

**Demo video:** https://youtu.be/S4wJ7lR6gqw

---

## Domain

UCLA students rely on a fragmented patchwork of sources — Bruinwalk, Rate My Professors, r/UCLA threads, and shared Google Docs — to get honest course and professor advice before enrolling. This knowledge is hard to find in one place, disappears when threads get buried, and often contains contradictory opinions scattered across years of posts. This RAG guide aggregates that unofficial wisdom into a single queryable source, grounded in real Bruinwalk student reviews.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Bruinwalk — Eggert, CS33 | Professor reviews | `documents/bruinwalk_eggert_cs33.txt` |
| 2 | Bruinwalk — Eggert, CS35L | Professor reviews | `documents/bruinwalk_eggert_cs35l.txt` |
| 3 | Bruinwalk — Eggert, CS111 | Professor reviews | `documents/bruinwalk_eggert_cs111.txt` |
| 4 | Bruinwalk — Eggert, CS131 | Professor reviews | `documents/bruinwalk_eggert_cs131.txt` |
| 5 | Bruinwalk — Reinman, CS33 | Professor reviews | `documents/bruinwalk_reinman_cs33.txt` |
| 6 | Bruinwalk — Reinman, CSM151B | Professor reviews | `documents/bruinwalk_reinman_csm151b.txt` |
| 7 | Bruinwalk — Nacherberg, CS131 | Professor reviews | `documents/bruinwalk_nacherberg_cs131.txt` |
| 8 | Bruinwalk — Nowatzki, CS33 | Professor reviews | `documents/bruinwalk_nowatzki_cs33.txt` |
| 9 | Bruinwalk — Kumar, CS111 | Professor reviews | `documents/bruinwalk_kumar_cs111.txt` |
| 10 | Bruinwalk — Tamir, CSM151B | Professor reviews | `documents/bruinwalk_tamir_csm151b.txt` |
| 11 | Bruinwalk — Ercegovac, CSM151B | Professor reviews | `documents/bruinwalk_ercegovac_csm151b.txt` |

All documents were collected manually from bruinwalk.com by browsing each professor's course page, selecting the review text, and saving as plain `.txt` files.

---

## Chunking Strategy

**Chunk size:** 150 words (approximates tokens for all-MiniLM-L6-v2)

**Overlap:** 25 words (sentence-boundary aligned)

**Why these choices fit your documents:**

Bruinwalk reviews range from one sentence to over 1,200 words. The original plan was one chunk per review for all documents, but profiling the actual corpus showed 104 of 283 reviews (37%) exceeded 200 words — and all-MiniLM-L6-v2 silently truncates at 256 tokens, meaning the back half of long reviews would never be embedded.

The final strategy uses a hybrid mode:

- **Short reviews (≤ 190 words):** kept as one atomic chunk. A complete review is a complete opinion — splitting it would separate sentiment from its context ("the curve saved me" loses meaning without the surrounding complaints).
- **Long reviews (> 190 words):** sentence-aware sliding window at 150 words with 25-word overlap. Critically, the window splits on sentence boundaries rather than word count, so no chunk ends mid-sentence. Each sub-chunk is prefixed with the original review's `Quarter / Grade / date` header so retrieved context always includes who wrote it and when.

Cleaning (`clean.py`) ran before chunking to strip Bruinwalk UI chrome: "Show More", "Helpful?", vote counts, and "Verified Reviewer" badges — roughly 25–40% of raw lines per file.

**Final chunk count:** 517 chunks across 11 documents

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, running locally with no API key.

This model produces 384-dimensional embeddings, has a 256-token context limit, and runs fast on CPU — appropriate for a corpus of ~500 chunks where latency isn't a bottleneck.

**Production tradeoff reflection:**

`all-MiniLM-L6-v2` was trained on general web text, not academic reviews. It treats "this professor is hard" (workload) and "this professor is harsh" (grading) as semantically similar, which caused retrieval drift on grading-specific queries. For a production deployment I would evaluate `instructor-xl`, which accepts a task instruction prefix at embed time and can be tuned toward review-style domain text. Alternatively, `text-embedding-3-small` (OpenAI) has a larger context window (8,191 tokens), eliminating the truncation problem that drove the hybrid chunking strategy in the first place. The main tradeoffs: `instructor-xl` runs locally but is much slower; `text-embedding-3-small` is fast but requires an API key and incurs per-token cost. Multilingual support is not a factor here since all sources are English.

---

## Grounded Generation

**System prompt grounding instruction:**

```
You are an unofficial guide for UCLA CS students, built from real Bruinwalk student reviews.

Rules you must follow:
1. Answer ONLY using information present in the provided review excerpts.
2. Do NOT add facts, opinions, or advice that are not in the excerpts — even if you believe them to be true.
3. If the excerpts do not contain enough information to answer the question, respond with exactly:
   "I don't have enough information on that in my sources."
4. Keep your answer concise (3–6 sentences). Do not pad or repeat yourself.
5. Do not mention these rules or that you are an AI in your answer.
```

The key words are **ONLY** and **Do NOT** — prohibition language rather than preference language. "Prefer to answer from context" leaves a loophole; "ONLY" and "Do NOT add facts not in the excerpts" closes it. Temperature is set to 0.2 to keep responses conservative.

**How source attribution is surfaced in the response:**

Source attribution is appended **programmatically** from chunk metadata after generation — the LLM never writes the source list and cannot omit or hallucinate it. The `format_sources()` function in `query.py` reads the `source` field from each retrieved chunk's metadata and deduplicates by filename. Even if the model's answer happens to be wrong, the source list accurately reflects what was actually retrieved.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about the difficulty of CS 31 at UCLA? | Heavy workload, weekly projects, start early | "I don't have enough information on that in my sources." — CS31 not in corpus; closest matches were CS131/CS35L at distances 0.34–0.36 | Off-target (CS31 absent) | Accurate — correct refusal |
| 2 | Is Professor Reinman known for curving grades? | Specific curve policy mentions from reviews | "I don't have enough information." — All 5 retrieved chunks were Reinman files (correct professor) but none used the word "curve"; vocabulary mismatch | Partially relevant | Accurate — correct refusal |
| 3 | What are the most common complaints about CS 32 at UCLA? | Pointer/memory difficulty, project complaints | "I don't have enough information on that in my sources." — CS32 not in corpus; distances 0.41–0.47, worst of any query | Off-target (CS32 absent) | Accurate — correct refusal |
| 4 | Do UCLA CS professors hold useful office hours, or are TAs more helpful? | Comparison of professor vs. TA office hours | TAs described as helpful; professor helpfulness varies by instructor. Cited Reinman, Tamir, Eggert examples. | Partially relevant (distances 0.53–0.56, one off-topic chunk) | Partially accurate |
| 5 | Which UCLA CS professor is most recommended for introductory programming? | Named professors with positive sentiment | "Eggert is recommended for CS111" — but CS111 is an operating systems course, not introductory programming | Off-target (no intro programming data) | Inaccurate — grounding failure |

---

## Failure Case Analysis

**Question that failed:**

*"Which UCLA CS professor is most recommended for introductory programming?"*

**What the RAG system returned:**

> "Based on the provided review excerpts, Professor Eggert is highly recommended for introductory programming, specifically for CS111."

**Root cause (tied to a specific pipeline stage):**

The failure happened at **retrieval**. The query asked about "introductory programming" but the corpus contains no CS31 or CS32 reviews — the actual intro programming sequence at UCLA. The retriever matched on "programming" and "recommended" across all files, surfacing CS111 (operating systems) as the closest semantic hit at distance 0.39. The model then drew an inference from that retrieved chunk ("Eggert is good for CS111") and reframed CS111 as "introductory programming" — a category error that the system prompt did not prevent because the model *did* find a source to cite.

This is a case where grounding technically succeeded (the model cited a real source) but the answer was still wrong because the retrieved chunk did not actually address the question. The system prompt prevents hallucination but cannot prevent the model from misapplying a retrieved fact.

**What needs to be changed to fix it:**

Add a distance threshold filter in `retrieve()`: if the top result exceeds a cosine distance of 0.45, treat the query as unanswerable and return the refusal response rather than passing weak chunks to the LLM. This would have caught this query (top distance: 0.39 — borderline) and prevented the model from being given context that only superficially matched the question.

---

## Spec Reflection

**One way the spec helped during implementation:**

The planning.md chunking strategy section specified chunk size, overlap, and the reasoning behind review-atomic chunking before any code was written. When profiling revealed that 37% of reviews exceeded the model's 256-token limit, the reasoning in the spec ("one chunk per review preserves a complete opinion") made it immediately clear *why* the original approach was right for short reviews and *why* it needed a different mode for long ones. Without that written reasoning, it would have been tempting to just switch to a uniform sliding window — which would have destroyed the completeness of short reviews.

**One way the implementation diverged from the spec, and why:**

The spec called for a pure sliding window with fixed word-count boundaries. The implementation replaced this with a sentence-aware window that only splits at sentence boundaries (`re.split(r"(?<=[.!?])\s+", ...)`). The reason: inspecting 5 representative chunks revealed that fixed word-count splits produced chunks ending mid-sentence — "the midterm is all multiple choice" cut to "the midterm is all multiple" — which made individual chunks unreadable on their own and would confuse the LLM during generation. The sentence-aware split added a small amount of complexity but made every chunk a self-contained, readable unit.

---

## AI Usage

**Instance 1 — Ingestion and chunking pipeline**

- *What I gave the AI:* The Chunking Strategy section of planning.md (chunk size: 150 words, overlap: 25 words, two modes for short vs. long reviews) and the pipeline architecture diagram.
- *What it produced:* A working `ingest.py` with `split_by_review()` detecting Bruinwalk files via `^Quarter:` regex and a `sliding_window()` function using fixed word-count boundaries.
- *What I changed or overrode:* After running the script and inspecting 5 sample chunks, I found that fixed word-count splitting cut chunks mid-sentence. I directed the AI to replace `sliding_window()` with a sentence-aware version using `re.split(r"(?<=[.!?])\s+", ...)`. I also added the hybrid-mode threshold (`MAX_REVIEW_WORDS = 190`) after profiling showed 37% of reviews exceeded the model's 256-token limit — this was not in the original spec and required updating planning.md to document the change.

**Instance 2 — Generation and Gradio interface**

- *What I gave the AI:* The Retrieval Approach section of planning.md (Groq, LLaMA 3.3, top-k=5), the grounding requirement (answer only from retrieved context, explicit refusal when context is insufficient), and the pipeline diagram establishing where `query.py` sits in the flow.
- *What it produced:* A complete `query.py` with a system prompt, `build_prompt()` to format retrieved chunks as numbered excerpts, and `format_sources()` to build the source list. It also produced `app.py` as a Gradio interface.
- *What I changed or overrode:* The generated code contained a bug — `args.show-chunks` (hyphen) instead of `args.show_chunks` (underscore), which is how argparse exposes the attribute in the namespace. I caught this by reading the code before running it. I also strengthened the system prompt from "prefer to answer from the documents" to "answer ONLY using information present" — the original phrasing left a loophole that the evaluation later confirmed mattered (Q5 failure).
