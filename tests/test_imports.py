"""Import every public subpackage. No network."""

from cadence.llm import MockClient


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


def test_mock_client_returns_canned_reply() -> None:
    client = MockClient()
    assert client.complete([{"role": "user", "content": "hi"}]) == "mock-ok"
