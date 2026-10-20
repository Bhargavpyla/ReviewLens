# ReviewLens Annotation Rules

Status: finalized after reviewing every row in the first 20-review pipeline preview on 2026-10-20.

The initial file was empty, so the rules were not finalized before predictions were viewed. This chronology limitation must be disclosed in the report. The later gold-labeling CSV must still be prediction-free.

## Labels

- Allowed aspects: `plot`, `characters`, `pacing`, `writing_style`, `ending`, `none`.
- Allowed sentiments for aspect-bearing units: `positive`, `negative`, `neutral`.
- `neutral` means a factual description or recap without an expressed opinion.
- When the aspect is `none`, leave `gold_sentiment` empty. Exclude these rows from aspect-level sentiment scoring.

## Annotation Unit

1. Label one sentence or clause per row.
2. Split a mixed sentence into clauses when the clauses express different aspects or sentiments.
3. Use one primary aspect per row unless the same indivisible clause clearly evaluates two aspects.
4. If a clause cannot be split without losing its meaning, use the dominant aspect and sentiment and record the ambiguity for error analysis.

## Aspect Decisions

1. Label the meaning of the text, not the presence of a keyword.
2. Use `plot` for opinions about the story, premise, mystery, twists, suspense, or events.
3. Use `characters` for opinions about characters, characterization, relationships, or character development.
4. Use `pacing` for speed, flow, dragging, rushing, chapter length, or engagement such as "hard to put down" and wanting to know what happens next.
5. Use `writing_style` for prose, dialogue, narration, wording, grammar, typos, proofreading, accents, or the author's technique.
6. Use `ending` only for an opinion about the ending or conclusion. Merely saying "read to the end" is not an ending opinion.
7. Use `none` for vague whole-book praise or criticism and content outside the five supported aspects. This includes illustrations, formatting, genre preference, shipping, seller, price, packaging, and physical condition.
8. A phrase such as "story book" that identifies a type of book is not automatically `plot`.
9. "The story dragged" is `pacing`, not `plot`. If a clause independently evaluates story content and speed, split it when possible.

## Sentiment Decisions

1. Label the reviewer's intended opinion, not isolated positive or negative words.
2. Plot recaps and factual descriptions are `neutral`, including facts containing words such as "kills", "dangerous", or "horrific".
3. Resolve negation compositionally: "not boring" is `positive`; "not interesting" is `negative`.
4. Resolve sarcasm by intended meaning rather than literal polarity.
5. Label mixed sentiment at clause level.

## Consistency Log

- "Kept me guessing" about a mystery is `plot`; "impossible to put down" or "kept me reading" is `pacing`.
- Criticism of culturally inaccurate dialogue or accents is `writing_style`.
- Opinions about illustrations remain `none` because visual design is outside the fixed schema.
- Non-narrative products in the Books category still use the fixed schema; unsupported content is `none`.
