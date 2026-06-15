# The Unofficial Guide — Project 1


## Domain

This system is an **unofficial student guide to eating at the University of Florida**. It covers practical, experience-based questions about UF dining: whether meal plans are worth it, which campus locations students like or avoid, where the food is a good value, and common complaints about quality, price, and convenience.

This knowledge is valuable because official UF dining pages explain locations, hours, meal-plan tiers, and policies, but they do not tell a student whether a plan is actually worth the money, which dining halls students think are repetitive, or what cheaper alternatives exist. That informal, opinion-based knowledge lives in scattered Reddit threads and word-of-mouth, which makes it hard to search. Combining the official pages with student discussion in one retrieval system lets a new student get both the facts and the lived-experience perspective in a single answer.

---

## Document Sources

The corpus mixes official UF dining information (for facts about plans, locations, and policies) with student-generated Reddit discussion (for opinions on value, quality, and convenience).

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Florida Fresh Dining: Who We Are | Official dining portal | https://new.dineoncampus.com/UF/who-we-are → `documents/01_uf_dining_who_we_are.txt` |
| 2 | Florida Fresh Dining: Hours & Locations | Official dining portal | https://dineoncampus.com/uf/hours-of-operation → `documents/02_uf_dining_locations.txt` |
| 3 | Navigating the Swamp: A Gator's Guide to UF Meal Plans | Unofficial guide | https://prked.com/post/navigating-the-swamp-a-gators-guide-to-uf-meal-plans → `documents/03_uf_meal_plans.txt` |
| 4 | UF Declining Balance | Official services page | https://businessservices.ufl.edu/services/dining/declining-balance/ → `documents/04_declining_balance.txt` |
| 5 | Florida Fresh Dining: FAQ | Official dining portal | https://new.dineoncampus.com/UF/faq → `documents/05_uf_dining_faq.txt` |
| 6 | Reddit: Best meal plan for freshman? | Reddit thread (r/ufl) | https://www.reddit.com/r/ufl/comments/1b5y8qy/best_meal_plan_for_freshman/ → `documents/06_reddit_best_meal_plan_freshman.txt` |
| 7 | Reddit: UF on campus food tier list | Reddit thread (r/ufl) | https://www.reddit.com/r/ufl/comments/xqubez/uf_on_campus_food_tier_list/ → `documents/07_reddit_food_tier_list.txt` |
| 8 | Reddit: Meal plans Freshman Fall 2025 | Reddit thread (r/ufl) | https://www.reddit.com/r/ufl/comments/1l77ghc/meal_plans_freshman_fall_2025/ → `documents/08_reddit_meal_plans_fall_2025.txt` |
| 9 | Reddit: Debating about Meal Plan or None | Reddit thread (r/ufl) | https://www.reddit.com/r/ufl/comments/14uywgx/debating_about_meal_plan_or_none/ → `documents/09_reddit_meal_plan_or_none.txt` |
| 10 | Reddit: Lunch? | Reddit thread (r/ufl) | https://www.reddit.com/r/ufl/comments/1mj9giu/lunch/ → `documents/10_reddit_lunch_recommendations.txt` |

---

## Chunking Strategy

**Chunk size:**
~500 characters per chunk (fixed-character sliding window, implemented in `chunk_text()` in `ingest.py`).

**Overlap:**
~100 characters of overlap between adjacent chunks (the window advances 400 characters each step).

**Why these choices fit your documents:**
The corpus mixes short, punchy student opinions with longer structured official text. A 500-character window is small enough to keep one dining-related idea in focus (a single recommendation, complaint, or policy detail) but large enough to preserve the surrounding context that gives an opinion meaning. The 100-character overlap reduces the chance that a recommendation and its reasoning (common in Reddit comments) get split across a boundary and retrieved only half-formed.

**Preprocessing before chunking:**
- Whitespace normalization: collapsed runs of spaces/tabs to single spaces and removed empty lines (`clean_text()`).
- Reddit artifact removal: stripped UI navigation noise (`Upvote`, `Downvote`, `Reply`, `Share`) and the standalone vote-count numbers from all five Reddit files, since these carry no dining meaning and would dilute embeddings.
- Header removal: the original `SOURCE:`/`URL:` lines at the top of each file were removed, so they do not appear in the final chunks. (Per-comment metadata such as usernames and `4y ago` timestamps was kept as lightweight context.)

**Final chunk count:**
259 chunks across the 10 documents.

**Limitations:**
Because the chunker uses a fixed character window rather than splitting on sentence or comment boundaries, some chunks begin or end mid-sentence (e.g., a chunk starting "l plan my freshmen year…"). This did not prevent retrieval from working, but it makes some retrieved chunks less readable and can feed partial context into generation. See the Failure Case Analysis for a concrete example.

---

## Sample Chunks

Five representative chunks from across the corpus (produced by `python ingest.py`), each labeled with its source document and chunk index:

**1. `01_uf_dining_who_we_are.txt` - chunk 0** (official)
> Who We Are
> Welcome to Florida Fresh Dining, where hospitality begins at the heart of the University of Florida! With over 45 dining locations, we pride ourselves on blending tradition with innovation, ensuring every Gator finds a culinary experience that resonates with their tastes and preferences. Our commitment to diversity and convenience drives us to constantly evolve, staying attuned to the latest food trends and culinary delights.

**2. `03_uf_meal_plans.txt` - chunk 5** (unofficial guide)
> Plans: For the Off-Campus & Independent Gators
> Commuter plans are perfect for students who live off-campus, have a more flexible schedule, or actually enjoy cooking for themselves. These plans are usually a set number of meals per semester (a "block" of meals) that you can use whenever you want. They also come with Flex Bucks, giving you the freedom...

**3. `05_uf_dining_faq.txt` - chunk 27** (official)
> $13.00 plus taxes / Dinner $13.50 plus taxes
> What is the door rate at Arredondo Cafe? $15.00 + 7.5% taxes
> When do the dining halls serve breakfast/lunch/dinner? Broward -Breakfast: 7:00 AM to 11:00 AM. L[unch]...

**4. `07_reddit_food_tier_list.txt` - chunk 4** (Reddit)
> ...campus since Spring '21. When did they get rid of Papa Johns?
> CloudWoww • 4y ago: They got rid of it earlier this may
> InsunLee • 4y ago (Alumni): Pollo tropical would be S if they didn't get my order wron[g]...

**5. `09_reddit_meal_plan_or_none.txt` - chunk 25** (Reddit)
> ...it'd be better to make my own breakfast, pay as I go, so I'm trying to find a balance here!
> LJkick • 3y ago (Graduate): If you purchase an all flex-bucks plan that would 100% work as well...

(Chunks 3 and 4 illustrate the fixed-window limitation noted above - they begin and end mid-sentence, e.g. "wron[g]".)

---

## Embedding Model

**Model used:**
`sentence-transformers/all-MiniLM-L6-v2`, loaded locally with `SentenceTransformer("all-MiniLM-L6-v2")`. Embeddings are stored in a local ChromaDB collection configured for cosine distance (`hnsw:space: cosine`), and retrieval returns the top 5 chunks per query along with `source` and `chunk_index` metadata. I chose this model because it runs locally with no API key, has no rate limits, is fast enough for an interactive demo, and produced strong results on both the official dining pages and the informal Reddit comments during testing.

**Production tradeoff reflection:**
For a real deployment where cost wasn't a constraint, I would weigh several tradeoffs against this default. **Accuracy on domain-specific text:** all-MiniLM-L6-v2 is a small, general model (384-dimensional embeddings); a larger or domain-tuned model might better capture student slang, sarcasm, and abbreviations. **Context length:** MiniLM truncates inputs around 256 tokens, which is fine for my ~500-character chunks but would force re-chunking for longer documents. **Latency vs. quality:** a hosted embedding API (e.g., OpenAI or Cohere) could improve retrieval quality but adds per-call cost, network latency, and a dependency on an external service. **Multilingual support:** not needed here, but would matter for a broader audience. I would A/B test a larger model against MiniLM on my five evaluation queries and only adopt it if the retrieval improvement justified the added compute and cost.

---

## Retrieval Test Results

Three queries run through `retrieve()` (top-5, cosine distance; lower is more similar). Output from `python retriever.py`.

**Query 1: "What do students say about whether UF freshman meal plans are worth it?"**

| Rank | Distance | Source (chunk) | Preview |
|------|----------|----------------|---------|
| 1 | 0.165 | `09_reddit_meal_plan_or_none.txt` (50) | "…best to just keep your money and spend it how you want rather than tying it up in UF systems…" |
| 2 | 0.168 | `06_reddit_best_meal_plan_freshman.txt` (0) | "Best meal plan for freshman? …Don't wanna jump the gun on an unlimited dining hall plan if it's not worth it." |
| 3 | 0.184 | `08_reddit_meal_plans_fall_2025.txt` (0) | "Meal plans Freshman Fall 2025 …Is it worth buying any meal plans as a freshman?" |
| 4 | 0.209 | `09_reddit_meal_plan_or_none.txt` (61) | "…[meal] plan my freshmen year and it was a great decision…" |
| 5 | 0.247 | `06_reddit_best_meal_plan_freshman.txt` (10) | "…take these comments with a grain of salt, because Reddit is very anti-meal plans…" |

*Why these are relevant:* All five chunks come from the three Reddit threads that explicitly debate freshman meal-plan value, and several are the original thread titles/questions ("Best meal plan for freshman?", "Is it worth buying any meal plans as a freshman?"). The very low distances (0.165–0.247) reflect near-verbatim overlap between the query phrasing and the source text. The set also captures *both* sides, pro ("a great decision") and con ("very anti-meal plans"), which is exactly what a "what do students say" question needs.

**Query 2: "What are common student complaints about UF campus dining?"**

| Rank | Distance | Source (chunk) | Preview |
|------|----------|----------------|---------|
| 1 | 0.330 | `06_reddit_best_meal_plan_freshman.txt` (4) | "The food variety at UF just isn't there and the dining halls don't have the best food…" |
| 2 | 0.410 | `06_reddit_best_meal_plan_freshman.txt` (3) | "…upvoting for the instant pot—i cooked in my dorm room most of the time…" |
| 3 | 0.414 | `03_uf_meal_plans.txt` (19) | "The Cons: Cost… The Food Can Get Repetitive…" |
| 4 | 0.434 | `08_reddit_meal_plans_fall_2025.txt` (3) | "…tiring to cook in a communal kitchen. The food is decent in the dining halls…" |
| 5 | 0.437 | `06_reddit_best_meal_plan_freshman.txt` (10) | "…Reddit is very anti-meal plans…" |

*Why these are relevant:* The top hit directly names two core complaints (lack of variety, mediocre dining-hall food), and chunk 3 of the official guide surfaces the explicit "Cons: Cost / Food Can Get Repetitive" list. This is useful because it frames the same complaints in neutral terms. Distances are higher than Query 1 (0.33–0.44) because "complaint" is an inferred concept rather than wording that appears literally in the threads, so the match is semantic rather than lexical. Chunk 2 (the instant-pot comment) is the weakest result. This is relevant only indirectly, as evidence that students cook to avoid dining.

**Query 3: "What are some alternatives to relying completely on a meal plan?"**

| Rank | Distance | Source (chunk) | Preview |
|------|----------|----------------|---------|
| 1 | 0.336 | `09_reddit_meal_plan_or_none.txt` (52) | "…if you like, we skip breakfast or get full off a bar but that swipe still has a cost…" |
| 2 | 0.371 | `03_uf_meal_plans.txt` (21) | "…If you're in an apartment-style dorm with a full kitchen, you might be ab[le to cook]…" |
| 3 | 0.379 | `09_reddit_meal_plan_or_none.txt` (34) | "…I got the block 65 which comes out to be about 5 meals a week… most amount of flex bux…" |
| 4 | 0.379 | `06_reddit_best_meal_plan_freshman.txt` (1) | "…it frees you up to buy groceries or food off campus… Just go cash/pay as you go." |
| 5 | 0.380 | `09_reddit_meal_plan_or_none.txt` (62) | "…I would consider the meal plan if I didn't have… kitchen access, or… a car. …learn how to cook…" |

*Why these are relevant:* Every chunk names a concrete alternative such as pay-as-you-go, buying groceries, cooking with kitchen access, or choosing a smaller block plan plus Flex Bucks which maps directly onto the question. The tight distance band (0.336–0.380) shows the corpus discusses this topic consistently across multiple threads rather than in one isolated place.

---

## Grounded Generation

**System prompt grounding instruction:**
Generation uses Groq's `llama-3.3-70b-versatile` at `temperature=0` with this exact system prompt (in `query.py`):

> "You are a grounded RAG assistant. Answer only using the provided context. Do not use outside knowledge. If the context does not contain enough information, say exactly: I don't have enough information in the provided documents to answer that."

Grounding is enforced structurally, not just suggested:
- The user message wraps the top-5 retrieved chunks in a labeled `Context:` block. Each chunk is prefixed with `[Source: <filename> | chunk <index>]` followed by the question and a final instruction to "Answer using only the context above."
- `temperature=0` keeps answers focused on the supplied text rather than creatively extrapolating.
- The prompt gives the model an explicit escape hatch (the fixed refusal sentence) so that "no answer" is a valid, expected output rather than something the model has to be coaxed into. This was verified with an out-of-scope question ("What is the best parking garage at UF?"), which correctly returned the refusal sentence even though some retrieved chunks mentioned parking in passing.

**How source attribution is surfaced in the response:**
Source attribution is **programmatically guaranteed**, not left to the LLM. After generation, `ask()` reads the `source` filename from each retrieved chunk's metadata and builds a de-duplicated, order-preserving list (`unique_sources()`). The returned dictionary contains `answer`, `sources` (the filename list), and `retrieved_chunks` (full text + source + chunk index + distance). The Gradio UI (`app.py`) displays the answer, the source list, and a "Retrieved chunks (evidence)" panel showing each chunk's source, index, distance, and a preview. A viewer can trace every answer back to specific documents regardless of what the model says in its prose.

---

## Example Responses

Verbatim output from `python query.py`. Source attribution (the `SOURCES:` line) is appended programmatically from chunk metadata, so it appears in every response.

**Example 1: in-scope (meal plan value):**
> **Question:** What do students say about whether UF freshman meal plans are worth it?
>
> **Answer:** According to the provided context, students have mixed opinions about whether UF freshman meal plans are worth it. Some students, such as Timely_Blessing_4804, mention that the main benefit of the meal plan is convenience, especially for those involved in sports, fraternities, or clubs. However, others, like Gatorboy-7 and Tan_batman, suggest that meal plans are not worth it due to limited options, lower food quality, and the potential for unused money to "expire"… [several students recommend buying groceries or paying as you go instead].
>
> **SOURCES:** `09_reddit_meal_plan_or_none.txt`, `06_reddit_best_meal_plan_freshman.txt`, `08_reddit_meal_plans_fall_2025.txt`

**Example 2: in-scope (official options):**
> **Question:** What official dining options or services does UF provide?
>
> **Answer:** Based on the provided context, UF provides: (1) Dining halls with "All You Care to Eat" Residential Plans, (2) the Food Hall at Gator Corner, (3) The Eatery, (4) Meal Swipes and Flex Bucks, and (5) Retail Equivalency Swipes. These options are managed by Florida Fresh Dining, the university's dining provider.
>
> **SOURCES:** `01_uf_dining_who_we_are.txt`, `03_uf_meal_plans.txt`, `05_uf_dining_faq.txt`, `09_reddit_meal_plan_or_none.txt`

**Example 3: out-of-scope (refusal):**
> **Question:** What is the best parking garage at UF?
>
> **Answer:** I don't have enough information in the provided documents to answer that.
>
> **SOURCES:** `03_uf_meal_plans.txt`, `09_reddit_meal_plan_or_none.txt`, `02_uf_dining_locations.txt`

The third example is the key grounding check: even though retrieval surfaced chunks that mention "garages for parking" in passing (the corpus is about dining, not parking), the model refused with the exact required sentence instead of fabricating a recommendation. (Note: the `SOURCES` list still shows the nearest retrieved files because attribution is mechanical. The *answer* correctly declines to use them.)

---

## Query Interface

The interface is a **Gradio web app** (`app.py`), launched with `python app.py` and served at `http://localhost:7860`.

**Input fields:**
- **Your question** - a textbox where the user types a natural-language question. Submitting works either by clicking the **Ask** button or pressing Enter in the box.

**Output fields:**
- **Answer** - the grounded response text from the LLM.
- **Sources (from retrieved documents)** - the de-duplicated list of source filenames the answer drew from.
- **Retrieved chunks (evidence)** - the raw top-5 chunks, each showing source, chunk index, cosine distance, and a ~200-character preview, so the user can inspect the evidence behind the answer.

**Sample interaction transcript:**
```
[User types in "Your question"]
  What do students say about whether UF freshman meal plans are worth it?

[Clicks "Ask"]

Answer:
  According to the provided context, students have mixed opinions… convenience is
  the main benefit (Timely_Blessing_4804), but others (Gatorboy-7, Tan_batman) say
  plans aren't worth it on price/quality and prefer groceries or pay-as-you-go.

Sources (from retrieved documents):
  • 09_reddit_meal_plan_or_none.txt
  • 06_reddit_best_meal_plan_freshman.txt
  • 08_reddit_meal_plans_fall_2025.txt

Retrieved chunks (evidence):
  [Rank 1] source=09_reddit_meal_plan_or_none.txt | chunk_index=50 | distance=0.165
      …best to just keep your money and spend it how you want rather than tying it up…
  [Rank 2] source=06_reddit_best_meal_plan_freshman.txt | chunk_index=0 | distance=0.168
      Best meal plan for freshman? …Don't wanna jump the gun on an unlimited plan…
  [ … ranks 3–5 … ]
```

---

## Evaluation Report

All five evaluation questions were run end-to-end through `ask()` (retrieval → grounded generation). Responses below are summarized; Full text was generated by running the five evaluation questions through `ask()` in `query.py` and can be reproduced through the Gradio app or a small script using `ask()`.

**Retrieval examples (distance scores, cosine):**
- *Q1: meal plans worth it:* top chunks from the Reddit meal-plan threads, distances ~0.154–0.26. Retrieval accurate.
- *Q2: common complaints:* top chunks on food variety, repetition, cost, and dining-hall quality, distances ~0.33–0.44. Retrieval accurate to partially accurate.
- *Q5: alternatives to a meal plan:* top chunks on groceries, pay-as-you-go, cooking, smaller plans, and kitchen access, distances ~0.279–0.301. Retrieval accurate.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about whether UF meal plans are worth it? | Convenient but not always the best value; some suggest smaller weekly plans or cooking/groceries instead of unlimited plans. | Reports mixed opinions: convenience is the main upside (esp. for busy freshmen), but several students say plans are a poor value on price/quality and prefer pay-as-you-go or groceries. Cites 09, 08, 06, 03. | Relevant (0.154–0.26, on-topic meal-plan threads) | **Accurate** |
| 2 | What are common student complaints about UF campus dining? | Food quality, limited weekend options, prices, long lines/inconvenience, few satisfying full-meal options. | Lists lack of variety, repetitive food, mediocre dining-hall quality, and high cost. Cites 06, 03, 08. | Relevant (0.33–0.44) | **Partially accurate** captures variety/repetition/cost complaints that appear in the retrieved chunks, but misses "limited weekend options" and "long lines," which weren't surfaced in the top 5. |
| 3 | What official dining options or services does UF provide? | Residential dining halls, campus locations, national/local brands, meal plans, declining balance / flex funds. | Lists All-You-Care-to-Eat residential plans, Food Hall at Gator Corner, The Eatery, meal swipes + Flex Bucks, and retail equivalency swipes, managed by Florida Fresh Dining. Cites 01, 03, 05, 09. | Relevant (0.36–0.43) | **Accurate** |
| 4 | What do students say is a good value for lunch on or near campus? | Dining halls can be a good lunch value (filling meals, drinks, salads, options) vs. buying out. | Says dining halls are a good lunch value (unlimited drinks, salads, variety) and names "Zania International Market's deli section" as having an excellent lunch special. Cites 10, 08, 03, 09. | Relevant (0.278–0.338) | **Partially accurate** see Failure Case Analysis. |
| 5 | What are some student-mentioned alternatives to relying completely on a meal plan? | Cooking, groceries, packing food, smaller plans, flex/declining balance, selected campus/off-campus spots. | Lists pay-as-you-go/cash, buying groceries and cooking, eating off campus, and using Flex Bucks to supplement a smaller plan. Cites 09, 06. | Relevant (0.279–0.301) | **Accurate** |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

Summary: 3 of 5 fully accurate, 2 partially accurate. The out-of-scope control question ("best parking garage at UF?") correctly returned the fixed refusal sentence rather than guessing, confirming the grounding instruction holds.

---

## Failure Case Analysis

**Question that failed:**
"What do students say is a good value for lunch on or near campus?" (Evaluation Q4)

**What the system returned:**
The system answered that dining halls are a good lunch value and added that "Zania International Market's deli section is mentioned as having an excellent lunch special." It stated this as a confident, factual recommendation.

**Root cause (tied to a specific pipeline stage):**
This is a **generation-stage grounding nuance combined with the fixed-window chunking limitation**. The underlying source line is a single, hedged off-hand comment: *"I think it's zania international market or similar, down south of the university, that has an excellent lunch special in their deli section and it is halal"* (`10_reddit_lunch_recommendations.txt`). The model's answer is technically grounded (the text exists in the retrieved chunk) but it **dropped the student's explicit uncertainty** ("I think… or similar") and the **"down south of the university" qualifier**, which means the place is *off* campus, not on it. So a tentative, single-person, location-ambiguous remark was elevated into a confident "good value near campus" recommendation. The fixed-character chunking contributes: because chunks are cut by character count rather than by comment, the hedging and the recommendation can land at different positions in the window, making it easier for the model to summarize the claim while losing its qualifiers.

**What you would change to fix it:**
1. **Prompt tuning:** instruct the model to preserve hedging and attribute opinions to a single commenter when only one source supports a claim (e.g., "If only one comment supports a point, present it as one student's opinion and keep any uncertainty they expressed").
2. **Chunking:** split Reddit files on comment boundaries instead of a fixed character count, so a comment's claim and its qualifiers stay together in one chunk.
3. **Retrieval filtering:** for "value/recommendation" questions, weight or require agreement across multiple chunks before presenting something as a general recommendation, rather than surfacing a lone remark.

---

## Spec Reflection

**One way the spec helped you during implementation:**
Writing the Chunking Strategy and Retrieval Approach sections in `planning.md` before coding gave each implementation step a concrete target. When I generated `ingest.py` and `retriever.py`, I could point directly at "500 characters, 100-character overlap, all-MiniLM-L6-v2, top-5, source + chunk_index metadata" instead of making those decisions mid-code. That made the generated code match my intent on the first pass and gave me fixed numbers to verify against (e.g., confirming chunks were ~500 chars and that metadata carried through to retrieval).

**One way your implementation diverged from the spec, and why:**
The original plan only specified whitespace cleaning. During Milestone 3 I noticed the Reddit files were full of UI artifacts (`Upvote`, `Downvote`, `Reply`, `Share`, and vote-count numbers), so I added a preprocessing step to strip them from all five Reddit files before chunking. This wasn't in the spec, but the artifacts added no dining meaning and were diluting the chunks; removing them together brought the corpus from 277 down to 259 chunks and made each chunk denser with real content. I kept per-comment metadata such as usernames and timestamps in place because it gives the model lightweight context for attribution.

---

## AI Usage


**Instance 1: Ingestion and chunking (Milestone 3)**

- *What I gave the AI:* My Domain, Documents, and Chunking Strategy sections from `planning.md`, plus a request for beginner-friendly, commented Python that loads `.txt` files, cleans whitespace, and chunks at ~500 characters with ~100-character overlap, preserving source filename and chunk index.
- *What it produced:* `ingest.py` with reusable `load_documents()`, `clean_text()`, `chunk_text()`, and `build_chunks()` functions plus a summary printout.
- *What I changed or overrode:* After inspecting sample chunks, I directed an extra preprocessing pass to strip Reddit navigation artifacts (Upvote/Downvote/Reply/Share and vote counts) that the initial cleaning didn't catch, which I had not originally planned for.

**Instance 2: Grounded generation and interface (Milestone 5)**

- *What I gave the AI:* My Retrieval Approach and the grounding requirement (answer only from retrieved context, refuse when context is insufficient, cite sources), plus the requirement that source attribution come from chunk metadata rather than from the LLM.
- *What it produced:* `query.py` (with `ask()`, a strict grounding system prompt, and programmatic source de-duplication) and `app.py` (a Gradio UI showing answer, sources, and retrieved chunks).
- *What I changed or overrode:* I pinned `gradio==6.9.0` after a newer version pulled an incompatible `huggingface-hub` that broke the embedding stack, and I verified the grounding actually held by testing an out-of-scope question to confirm the system refused rather than guessing.
