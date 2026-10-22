# Phase 6 spec: Code / Web UI stream (Continue.dev tasks)

Branch: `phase-6-codeweb`, created from `phase-5-loop` (or `main` once Phase 5 is merged).
Read first: `.continue/rules/cadence.md`, `cadence/tasks/base.py`, `cadence/tasks/text/tasks.py` (pattern to copy),
`cadence/tasks/toy.py`.

Do the tasks in order: A (data) -> B (validators) -> C (task + registry) -> D (plots, independent) -> E (edge tests).
Commit after each task. Run `python -m pytest` after each.

---------------------------------------------------------------------------------------------------------------

## Task A: `data/tasks/code_web.jsonl` (30 task specs)

One JSON object per line, UTF-8, no trailing commas, no blank lines.

### Record schema (all keys required)
| key | type | rule |
|---|---|---|
| `id` | string | `cw-001` ... `cw-030`, unique, zero-padded |
| `category` | string | one of `form`, `tool`, `dashboard`, `game`, `content`, `widget` (5 specs each) |
| `brief` | string | 30 to 90 words. See "Brief rules" |
| `requires.selectors` | list[string] | 3 to 8 items, simple selectors only (grammar below) |
| `requires.text` | list[string] | 1 to 4 visible strings the page must show |
| `requires.scripts` | bool | true if the page needs JavaScript behaviour |
| `requires.responsive` | bool | true if the brief demands a phone-friendly layout |
| `forbid_external` | bool | always `true` in this dataset |
| `behaviours` | list[string] | 2 to 4 human-readable behaviours (for a future judge and for humans; not machine checked) |

### Selector grammar (the validator only supports this; nothing else is allowed)
One compound selector, no spaces, no combinators (`>`, `+`, `~`, descendant), no pseudo-classes. Exactly one of:
`tag`, `#id`, `.class`, `tag#id`, `tag.class`, `tag[attr]`, `tag[attr=value]`, `tag#id[attr=value]`.
`tag` is lowercase letters/digits. `id`, `class`, `attr`, `value` match `[A-Za-z0-9_-]+`.
Regex the tests will enforce on every selector:
`^[a-z][a-z0-9]*(#[A-Za-z0-9_-]+)?(\.[A-Za-z0-9_-]+)?(\[[A-Za-z0-9_-]+(=[A-Za-z0-9_-]+)?\])?$|^[#.][A-Za-z0-9_-]+$`

### Brief rules (this is the most important part)
The model that reads the brief never sees `requires`. So **the brief must state, verbatim, every id, class,
attribute value and required text** that `requires` checks. Write ids/classes in backticks and required text in double
quotes. Example: "...a button with id `submit-btn` labelled "Create account"...". A test checks this.
Other rules:
- One self-contained HTML file. Never mention external libraries, CDNs, fonts, images, APIs or network access.
- Describe behaviour concretely (what happens on click, what is shown on error). No vague words like "nice" or "modern".
- Vary difficulty: about 10 simple static pages (`scripts: false`), the rest need JS. At least 20 specs have `scripts: true`.
  At least 10 specs have `responsive: true` (brief then says it must work on a phone screen around 360px wide).
- No two specs may share the same brief or the same id.

### Three complete example records (copy this style, do not reuse them)
```json
{"id":"cw-001","category":"form","brief":"Build a single-file sign-up page. It contains a form with id `signup-form`, an email input with id `email` (type email), a password input with id `password` (type password), and a submit button with id `submit-btn` labelled \"Create account\". When the form is submitted, use JavaScript to check that the password has at least 8 characters; if not, show a message in an element with id `error` and do not submit. The page must work on a phone screen about 360px wide.","requires":{"selectors":["form#signup-form","input#email","input#password","button#submit-btn","#error"],"text":["Create account"],"scripts":true,"responsive":true},"forbid_external":true,"behaviours":["A password shorter than 8 characters shows a message in #error and the form is not submitted","The layout stays usable at 360px width"]}
{"id":"cw-002","category":"tool","brief":"Build a tip calculator with the heading \"Tip calculator\". It has number inputs with ids `bill`, `tip-percent` and `people`, and a button with id `calc-btn` labelled \"Calculate\". Clicking the button writes the tip amount and the per-person total into an element with id `result`. Treat empty or negative inputs as invalid and write \"Enter valid numbers\" into `result` instead.","requires":{"selectors":["input#bill","input#tip-percent","input#people","button#calc-btn","#result"],"text":["Tip calculator","Calculate"],"scripts":true,"responsive":false},"forbid_external":true,"behaviours":["Bill 100, tip 10, people 2 shows a tip of 10 and 55 per person","Empty or negative inputs show \"Enter valid numbers\""]}
{"id":"cw-003","category":"content","brief":"Build a static pricing page. It has an `h1` reading \"Pricing\" and a `section` with class `plans` that holds three `article` elements with class `plan`. The plans are named \"Free\", \"Pro\" and \"Team\"; each has a price and a list of three features. Use only HTML and CSS, no JavaScript. The layout must work on a phone screen about 360px wide, with the plans stacked vertically.","requires":{"selectors":["h1","section.plans","article.plan","meta[name=viewport]"],"text":["Pricing","Free","Pro","Team"],"scripts":false,"responsive":true},"forbid_external":true,"behaviours":["Three plan cards are visible","Cards stack in one column on a narrow screen"]}
```
(`meta[name=viewport]` is allowed as a selector. For `responsive: true` specs the validator adds a separate viewport
check, so you need not list it, but listing it is fine.)

### Task A acceptance
Create `tests/test_codeweb_specs.py` that loads the file and asserts: exactly 30 records; schema/keys/types as above;
ids unique and well-formed; 5 per category; word counts; selector regex; every `#id`, `.class` and `[attr=value]`
token of every selector and every `requires.text` string appears verbatim in the brief; counts of `scripts: true`
(>= 20) and `responsive: true` (>= 10); no brief contains `http`, `cdn`, `fetch(` or `import `.
(After Task B exists, add the satisfiability test described under Task B.)

---------------------------------------------------------------------------------------------------------------

## Task B: validators in `cadence/tasks/code_web/validators.py`

Use BeautifulSoup with the `html5lib` parser (both already installed). Pure functions, no network.

### Public API
```python
class Check(BaseModel):            # pydantic
    name: str
    weight: float
    passed: bool
    detail: str = ""

class ValidationResult(BaseModel):
    score: float                   # weighted pass fraction in [0, 1]
    checks: list[Check]
    node_checked: bool             # True only if `node --check` actually ran

def extract_html(text: str) -> str: ...
def validate_page(html: str, spec: dict, *, use_node: bool = True) -> ValidationResult: ...
```
`spec` is one record from `code_web.jsonl` (read `spec["requires"]` and `spec["forbid_external"]`; tolerate missing
optional keys with defaults: `selectors=[]`, `text=[]`, `scripts=False`, `responsive=False`, `forbid_external=False`).

### `extract_html(text)`
Model replies often wrap HTML in markdown. In this priority order:
1. First fenced block whose info string is `html` (case-insensitive).
2. First fenced block (any info string) whose body contains `<html` or `<!doctype` (case-insensitive).
3. From the first `<!doctype` or `<html` (case-insensitive) to the last `</html>` inclusive (to the end if there is
   no `</html>`).
4. Otherwise `text.strip()`.
Always return the stripped result.

### Checks, names and weights (these exact names and weights; tests pin them)
| name | weight | included when | passes when |
|---|---|---|---|
| `parses` | 1.0 | always | `<html` and `</html>` both appear (case-insensitive) and the parsed `<body>` has text or at least one child element. A missing `</html>` means the output was truncated: fail |
| `doctype` | 0.5 | always | stripped html starts with `<!doctype html` (case-insensitive) |
| `selector:<sel>` | 1.0 each | one per `requires.selectors` item | `soup.select(sel)` is non-empty. An invalid selector (exception) = fail with the error in `detail` |
| `text:<str>` | 1.0 each | one per `requires.text` item | the string occurs, case-insensitive with whitespace collapsed, in the VISIBLE text |
| `scripts` | 1.0 | `requires.scripts` is true | at least one inline `<script>` (no `src`, JavaScript type, non-empty body) |
| `js_syntax` | 2.0 | the page has inline JS scripts AND `use_node` AND `node` is available | every inline script passes `node --check` |
| `no_external` | 1.0 | `forbid_external` is true | no external resource, see below |
| `size` | 0.5 | always | `len(html) <= 60000` characters |
| `viewport` | 0.5 | `requires.responsive` is true | a `<meta name="viewport">` exists |

Visible text = text of the parsed document with `script`, `style`, `template`, `noscript` and `head` removed, PLUS the
values of the attributes `placeholder`, `value`, `alt`, `aria-label` and `title` on any element. Collapse whitespace.

External resource = an attribute value starting with `http://`, `https://` or `//` on `script[src]`, `link[href]`,
`img[src]`, `iframe[src]`, `source[src]`, `video[src]`, `audio[src]`; or `url(` followed by `http`/`//`, or
`@import` with `http`/`//`, inside a `<style>` element or any `style` attribute. Plain `<a href="https://...">`
links are allowed.

JavaScript scripts: skip `<script>` with a `type` that is not empty, `text/javascript`, `application/javascript` or
`module`. Write each script to a temporary `.js` file (`.mjs` when `type="module"`), run `node --check <file>` with
`subprocess.run(..., timeout=10, capture_output=True)`, delete the files afterwards. Find node with `shutil.which("node")`
(import `shutil` as a module so tests can monkeypatch `cadence.tasks.code_web.validators.shutil.which`). A timeout or
OSError counts as a failed `js_syntax` check with the reason in `detail`.

### Score
`score = sum(weight for passed checks) / sum(weight of all included checks)`. If node is unavailable or `use_node`
is false, the `js_syntax` check is simply not included (so the score is not penalised) and `node_checked` is false.
Never raise on bad input: empty string, binary garbage and very long input all return a result.
Deterministic: same input gives an equal `ValidationResult`.

Worked example the tests must reproduce: spec of cw-001 above, page satisfying everything except the doctype, with
`use_node=False`: included weights = parses 1 + doctype 0.5 + 5 selectors + 1 text + scripts 1 + no_external 1 +
size 0.5 + viewport 0.5 = 10.5; passed = 10.0; score = 10 / 10.5.

### Task B acceptance: `tests/test_codeweb_validators.py` (at least these cases)
1. `extract_html`: html fence, generic fence containing `<html`, raw text with prose around a document, plain text
   with no html, fence in upper case.
2. Perfect page: score 1.0 with `use_node=False`; every check passed.
3. The worked example above (missing doctype): `score == pytest.approx(10 / 10.5)`.
4. Missing selector fails exactly `selector:<sel>`; invalid selector (e.g. `form >`) fails with a detail and does not raise.
5. Required text hidden inside `<script>`, `<style>` or `<head><title>` does NOT count; text in `placeholder` or `value` does.
6. External `<script src="https://...">`, `<link href="//...">`, and CSS `url(https://...)` fail `no_external`;
   `<a href="https://example.com">` passes.
7. Missing or empty inline script fails `scripts`; a script with `src` does not count.
8. Truncated page (no `</html>`) fails `parses`.
9. Page over 60000 characters fails `size`.
10. `responsive: true` without a viewport meta fails `viewport`; with it passes.
11. Empty string and `"\x00\x01garbage"` do not raise; empty string scores below 0.25.
12. Determinism: two calls give equal `model_dump()`.
13. JS syntax (mark `skipif(shutil.which("node") is None)`): valid script passes `js_syntax`; `var x = ;` fails it and
    lowers the score; `node_checked` is true.
14. Node missing (monkeypatch `shutil.which` to return None): no `js_syntax` check present, `node_checked` false,
    perfect page still scores 1.0.
15. Satisfiability of the dataset: add to `tests/test_codeweb_specs.py` a helper that builds a minimal page from a spec
    (one element per selector built from the grammar, one `<p>` per required text, a viewport meta, a doctype, and one
    inline `<script>var a = 1;</script>`) and asserts `validate_page(page, spec, use_node=False).score == 1.0` for all
    30 specs.

---------------------------------------------------------------------------------------------------------------

## Task C: `CodeWebTask` in `cadence/tasks/code_web/task.py`, plus registry

```python
class CodeWebTask(Task):
    name = "code_web"
    stream = "code_web"
    def __init__(self, records: Sequence[dict] | None = None, path: Path | None = None) -> None: ...
```
- Records come from `records`, else `path`, else `ROOT / "data" / "tasks" / "code_web.jsonl"` (use `cadence.config.ROOT`).
  Missing file: raise `FileNotFoundError` with a helpful message (same style as `TextTask`).
- Example: `id=rec["id"]`, `input={"brief": rec["brief"]}`, `reference=None`, `meta={"spec": rec, "category": rec["category"]}`.
- `load(split, n, seed)`: `select(self._all(), split, n, seed)` from `cadence.tasks.splits` (same as TextTask).
  Duplicate ids raise `ValueError`.
- `format_input(ex)`: returns the brief string unchanged.
- `seed_prompt()`: "You are a front-end engineer. Implement the brief as ONE self-contained HTML file with inline CSS
  and JavaScript and no external resources. Reply with only the complete HTML document, starting with <!DOCTYPE html>."
- `build_request(template, ex)`: same as the base class, but add `params={"max_tokens": 4096}`.
- `automatic_score(ex, out)`: `validate_page(extract_html(out.text or ""), ex.meta["spec"]).score` (node enabled).
  It never returns `None` in this phase, so the cascade makes no judge calls for this task.
- `rubric(ex)` returns `None`. Leave `calibration_rubric`, `degraded_prompt`, `corrupt_output` at their defaults.
- Export `CodeWebTask` from `cadence/tasks/code_web/__init__.py`, and append `"code_web": CodeWebTask` to `TASKS`
  in `cadence/tasks/registry.py` (append only; another session also adds lines there).

Acceptance: `tests/test_codeweb_task.py` using `CodeWebTask(records=[...])` with 10 tiny in-test records:
loads and splits deterministically and disjointly (opt/heldout/calib cover every id once); example input has the
brief and `reference is None`; `build_request` puts the template in the system message, the brief in the user message,
and `params["max_tokens"] == 4096`; `automatic_score` returns 1.0 for a perfect page and a lower value for a broken
one; missing data file raises `FileNotFoundError`; duplicate ids raise `ValueError`; `get_task("code_web")` works;
and an end-to-end check: `Evaluator`/`optimize` from `cadence.orchestrator` runs 1 generation on a `CodeWebTask`
with a `MockClient` whose `generator` script returns a perfect page (no network).
Manual check for the human afterwards: `python -m cadence.cli run --stream code_web --task code_web --mock --no-mlflow --max-gen 1`.

---------------------------------------------------------------------------------------------------------------

## Task D (auxiliary, independent): Pareto plots in `cadence/benchmark/plots.py`

Another session builds the rest of `cadence/benchmark/` and will call this module, so keep exactly this interface.
Create `cadence/benchmark/__init__.py` content untouched if it already exists.

```python
def load_members(archive_json: Path) -> list[Evaluated]: ...             # reads runs/<id>/archive.json (a JSON list of Evaluated)
def plot_pareto_3d(members: Sequence[Evaluated], out_path: Path) -> Path: ...
def plot_pareto_2d(members: Sequence[Evaluated], out_prefix: Path) -> list[Path]: ...
```
- Use matplotlib with the non-interactive `Agg` backend (set it inside the module with `matplotlib.use("Agg")`
  before importing pyplot). No display needed.
- `plot_pareto_3d`: scatter, x = cost per 1k runs, y = length in tokens, z = quality; label axes with units; annotate
  each point with a short rank number (1 = best quality); title includes the number of members; saves a PNG at
  `out_path` (create parent directories) and returns it.
- `plot_pareto_2d`: three PNGs `<prefix>_quality_cost.png`, `<prefix>_quality_length.png`, `<prefix>_cost_length.png`;
  returns the three paths in that order.
- Must work for 0 members (draw an empty axes with the text "no members"), 1 member, and members with identical
  values. Must not crash if cost is 0 for every member.
- Tests (`tests/test_plots.py`): files are created and non-empty and begin with the PNG signature `\x89PNG`;
  empty, single, duplicate and all-zero-cost inputs do not raise; `load_members` round-trips an archive written by
  `ParetoArchive.to_json()`.

## Task E (auxiliary): edge-case tests for existing code (add tests only; change no source)

Add `tests/test_edge_cases.py` covering, with `MockClient` / pure functions only:
- `ParetoArchive`: empty archive hypervolume is 0; adding the same candidate id twice replaces it; adding an
  identical-objective candidate returns False; `max_size=1` keeps exactly one member; JSON round-trip preserves members.
- `hypervolume`: single point; points outside the reference point contribute 0; duplicate points.
- `select_parents` and `minibatch_for` (from `cadence.orchestrator.loop`): `k` larger than the pool, `size` larger than
  the pool, empty pool raises `ValueError` for `minibatch_for`.
- `extract_template` (from `cadence.orchestrator.generator`): nested tags, missing closing tag, empty reply. Run the
  function first and pin its CURRENT behaviour in the test (e.g. a missing closing tag returns the whole text, tag
  included); if the behaviour looks wrong, say so in your final message but do not change the source.
- `optimize` resume: a run directory with a deleted `archive.json` and `resume=True` raises a clear error instead of
  silently restarting (if it currently does not, report that as a bug in your final message; do NOT change source).

---------------------------------------------------------------------------------------------------------------

## Definition of done for the whole brief
- `python -m pytest` green (the new tests included), under 60 seconds, no network.
- `python -m cadence.cli run --stream code_web --task code_web --mock --no-mlflow --max-gen 1 --minibatch 4` finishes
  and prints a Pareto menu.
- Branch pushed, PR opened against `phase-5-loop` (or `main` if Phase 5 is merged). Final message lists the commands
  run and the last pytest summary line.
