# ReviewLens Annotation Rules

Status: finalized after reviewing the first 20-review pipeline preview on 2026-10-20.
The initial file was empty, so these rules were not completed before predictions were seen. This chronology must be disclosed in the report. The later gold-labeling CSV must still be prediction-free.

## Labels

- Allowed aspects: `plot`, `characters`, `pacing`, `writing_style`, `ending`, `none`.
- Allowed sentiments: `positive`, `negative`, `neutral`.
- `neutral` means factual description or recap without an expressed opinion.
- For `none`, sentiment is ignored during aspect-level sentiment evaluation.

## Aspect Decisions

1. Label the meaning of the text, not the presence of a keyword.
2. Use one primary aspect per annotation unit. Split a mixed sentence into clauses when each clause expresses a different aspect.
3. Use `plot` for opinions about the story, premise, mystery, twists, suspense, or events.
4. Use `characters` for opinions about characters, characterization, relationships, or character development.
5. Use `pacing` for speed, flow, dragging, rushing, chapter length, engagement such as “hard to put down,” or wanting to know what happens next.
6. Use `writing_style` for prose, dialogue, narration, wording, grammar, typos, proofreading, accents, or the author's technique.
7. Use `ending` only for an opinion about the ending or conclusion. Merely saying “read to the end” is not an ending opinion.
8. Use `none` for vague whole-book praise or criticism, illustrations, formatting, genre preference, shipping, seller, price, packaging, and physical condition.
9. A phrase such as “story book” that identifies a type of book is not automatically `plot`.
10. “The story dragged” is `pacing`, not `plot`. If a clause independently evaluates both story content and speed, split it when possible.

## Sentiment Decisions

1. Label the reviewer’s intended opinion, not isolated positive or negative words.
2. Plot recaps and factual descriptions are `neutral`, including facts containing words such as “kills,” “dangerous,” or “horrific.”
3. Resolve negation compositionally: “not boring” is `positive`; “not interesting” is `negative`.
4. Resolve sarcasm by intended meaning rather than literal polarity.
5. Label mixed sentiment at clause level. If a clause cannot be split without losing meaning, use the dominant expressed sentiment and record the ambiguity.

## Consistency Log

- “Kept me guessing” about a mystery is `plot`; “impossible to put down” or “kept me reading” is `pacing`.
- Criticism of culturally inaccurate dialogue or accents is `writing_style`.
- Opinions about illustrations remain `none` because visual design is outside the fixed schema.
- Non-narrative products in the Books category still use the fixed schema; content outside those five aspects is `none`.
