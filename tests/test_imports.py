"""Import every public subpackage. No network."""

from cadence.llm import MockClient
from cadence.schemas import GenRequest


def test_import_all_subpackages() -> None:
    import cadence.benchmark
    import cadence.cascade
    import cadence.cost
    import cadence.judge
    import cadence.llm
    import cadence.orchestrator
    import cadence.pareto
    import cadence.tasks
    import cadence.tasks.audio
    import cadence.tasks.code_web
    import cadence.tasks.image
    import cadence.tasks.text
    import cadence.tracking
    import cadence.validator

    assert cadence.tasks.Task is not None


def test_mock_client_returns_text_and_usage() -> None:
    text, usage = MockClient().complete(GenRequest(role="generator", messages=[{"role": "user", "content": "hi"}]))
    assert text.startswith("mock:generator:")
    assert usage.usd == 0.0
