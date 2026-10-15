import pytest

from cadence.pareto import ParetoArchive
from cadence.cost import template_length
from cadence.schemas import Candidate, Evaluated, Objectives
from cadence.tasks import get_task
from cadence.tasks.toy import ToyTask, keywords_present, p_correct

TEMPLATES = [
    "Answer the question.",
    "Answer the question. Be concise.",
    "Answer the question. Think step by step.",
    "Answer the question. Think step by step, show your work, and double-check.",
    "Answer the question. Think step by step, show your work, double-check, and cite the context.",
]


def evaluate(task: ToyTask, template: str, n: int = 200) -> Objectives:
    exs = task.load("opt", n=n)
    outs = [task.generate(None, task.build_request(template, e)) for e in exs]
    q = sum(task.automatic_score(e, o) or 0 for e, o in zip(exs, outs)) / len(exs)
    avg_usd = sum(o.usage.usd for o in outs) / len(outs)
    return Objectives(quality=q, cost_usd_per_1k=1000 * avg_usd, length_tokens=template_length(template))


def test_keywords_and_probability() -> None:
    assert keywords_present("Think STEP BY STEP") == ["step by step"]
    assert p_correct("x") == pytest.approx(0.25)
    assert p_correct(TEMPLATES[-1]) == pytest.approx(0.80)  # 0.25 + 0.20 + 0.10 + 0.15 + 0.10


def test_generation_is_deterministic_and_offline() -> None:
    task = ToyTask()
    ex = task.load("opt", n=1)[0]
    req = task.build_request(TEMPLATES[2], ex)
    assert task.generate(None, req) == task.generate(None, req)


def test_quality_cost_and_length_rise_with_helpful_phrases() -> None:
    task = ToyTask()
    plain, steps, everything = (evaluate(task, TEMPLATES[i]) for i in (0, 2, 4))
    assert plain.quality < steps.quality < everything.quality
    assert plain.cost_usd_per_1k < steps.cost_usd_per_1k < everything.cost_usd_per_1k
    assert plain.length_tokens < steps.length_tokens < everything.length_tokens
    assert evaluate(task, TEMPLATES[1]).cost_usd_per_1k < plain.cost_usd_per_1k  # 'be concise' is cheaper


def test_toy_task_produces_a_real_tradeoff_front() -> None:
    task = ToyTask()
    arch = ParetoArchive()
    for i, t in enumerate(TEMPLATES):
        arch.add(Evaluated(candidate=Candidate(id=f"c{i}", template=t), objectives=evaluate(task, t), n_examples=200))
    assert len(arch) >= 3  # cheap-and-weak through expensive-and-strong all survive


def test_toy_splits_and_registry() -> None:
    task = get_task("toy")
    opt, held = {e.id for e in task.load("opt")}, {e.id for e in task.load("heldout")}
    assert opt and held and opt.isdisjoint(held)
