"""Candidate generation: reflective rewrite, mutation (shorten / strengthen) and crossover.

All proposals are plain `client.complete` calls on the "reflector" role, so they pass through the cache, retries and
the BudgetGuard like every other call. DSPy is deliberately NOT used here: its optimizers run their own LM loop and
would bypass the budget guard. If a DSPy signature is wanted later, wrap it behind `Generator.propose` and route its
LM through `cadence.llm`.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from typing import Any

from cadence.cost import count_tokens
from cadence.orchestrator.evaluate import Trace
from cadence.schemas import Candidate, Evaluated, GenRequest

OPERATORS = ("reflect", "shorten", "crossover", "strengthen")
MAX_TEMPLATE_TOKENS = 400

_SYSTEM = (
    "You are an expert prompt engineer improving a SYSTEM PROMPT TEMPLATE for a language model. The template is "
    "sent as the system message; each task example is sent as the user message. Optimise three things at once: "
    "answer quality (higher), cost per call (lower: shorter prompts and shorter answers cost less) and the template's "
    "own length (shorter). Never add the answers to specific examples. Reply with ONLY the new template inside "
    "<template></template> tags."
)

_TASKS = {
    "reflect": "OPERATOR: reflect\nRewrite the parent template to fix the failures shown below, keeping it concise.",
    "shorten": "OPERATOR: shorten\nRewrite the parent template to be clearly shorter and cheaper while keeping what makes it work.",
    "strengthen": "OPERATOR: strengthen\nRewrite the parent template to improve answer quality by adding the single most useful instruction.",
    "crossover": "OPERATOR: crossover\nCombine the best ideas of parent A and parent B into one coherent template.",
}

_TAG = re.compile(r"<template>(.*?)</template>", re.S | re.I)
_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```$", re.S)


def extract_template(text: str) -> str:
    """Pull the template out of a reflector reply: <template> tags, else a code fence, else the whole text."""
    m = _TAG.search(text)
    if m:
        body = m.group(1)
    else:  # an unclosed tag (truncated reply): keep what follows it, never the tag text itself
        i = text.lower().find("<template>")
        body = text[i + len("<template>"):] if i != -1 else text
    body = body.strip()
    f = _FENCE.match(body)
    return (f.group(1) if f else body).strip()


def normalize(template: str) -> str:
    return " ".join(template.lower().split())


def template_hash(template: str) -> str:
    return hashlib.sha256(normalize(template).encode()).hexdigest()[:8]


def _fmt_traces(traces: Sequence[Trace]) -> str:
    if not traces:
        return "(no failure examples available)"
    return "\n".join(
        f"- input: {t.input}\n  output: {t.output}\n  score: {t.quality:.2f}" for t in traces
    )


class Generator:
    def __init__(self, client: Any, *, role: str = "reflector", seed: int = 0, task_hint: str = "") -> None:
        self.client, self.role, self.seed, self.task_hint = client, role, seed, task_hint

    def _request(self, op: str, parents: Sequence[Evaluated], traces: Sequence[Trace]) -> GenRequest:
        a = parents[0]
        lines = [_TASKS[op], f"TASK: {self.task_hint}" if self.task_hint else ""]
        if op == "crossover":
            b = parents[1]
            lines += [f"PARENT A (quality {a.objectives.quality:.2f}):\n<<<\n{a.candidate.template}\n>>>",
                      f"PARENT B (quality {b.objectives.quality:.2f}):\n<<<\n{b.candidate.template}\n>>>"]
        else:
            o = a.objectives
            lines.append(f"PARENT TEMPLATE (quality {o.quality:.2f}, ${o.cost_usd_per_1k:.4f}/1k, "
                         f"{o.length_tokens} tokens):\n<<<\n{a.candidate.template}\n>>>")
            if op in ("reflect", "strengthen"):
                lines.append("LOWEST-SCORING EXAMPLES:\n" + _fmt_traces(traces))
        return GenRequest(role=self.role, messages=[{"role": "system", "content": _SYSTEM},
                                                    {"role": "user", "content": "\n\n".join(x for x in lines if x)}],
                          params={"max_tokens": 700})  # no temperature: Gemini 3+ deprecates it; the default already varies replies

    def schedule(self, n: int, n_parents: int, gen: int) -> list[str]:
        """Operator per child; rotates with the generation so different gens start differently. Crossover needs 2 parents."""
        ops = [o for o in OPERATORS if o != "crossover" or n_parents >= 2]
        off = random.Random(self.seed * 7919 + gen).randrange(len(ops))
        return [ops[(off + i) % len(ops)] for i in range(n)]

    def propose(self, parents: Sequence[Evaluated], traces: dict[str, Sequence[Trace]], n: int, gen: int,
                seen: set[str]) -> list[Candidate]:
        """Up to n new, distinct, valid candidates. `seen` (normalised-template hashes) is updated in place."""
        if not parents:
            raise ValueError("propose needs at least one parent")
        rng = random.Random(self.seed * 104729 + gen)
        out: list[Candidate] = []
        for k, op in enumerate(self.schedule(n, len(parents), gen)):
            first = rng.choice(list(parents))
            if op == "crossover":
                others = [p for p in parents if p.candidate.id != first.candidate.id]
                chosen = [first, rng.choice(others)]
            else:
                chosen = [first]
            req = self._request(op, chosen, traces.get(first.candidate.id, []))
            text, _usage = self.client.complete(req)  # BudgetExceeded propagates to the loop
            tpl = extract_template(text)
            h = template_hash(tpl)
            if not tpl or h in seen or count_tokens(tpl) > MAX_TEMPLATE_TOKENS:
                continue
            seen.add(h)
            out.append(Candidate(id=f"g{gen}c{k}-{h}", template=tpl,
                                 parent_ids=[p.candidate.id for p in chosen], generation=gen))
        return out
