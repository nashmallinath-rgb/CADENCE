"""Optimisation loop: candidate generation, evaluation, and the closed loop itself."""
from cadence.orchestrator.evaluate import EvalResult, EvaluationFailed, Evaluator, Trace, worst_traces
from cadence.orchestrator.generator import Generator, extract_template, template_hash
from cadence.orchestrator.loop import LoopSettings, OptimizationResult, optimize, pareto_menu

__all__ = ["EvalResult", "EvaluationFailed", "Evaluator", "Generator", "LoopSettings", "OptimizationResult", "Trace",
           "extract_template", "optimize", "pareto_menu", "template_hash", "worst_traces"]
