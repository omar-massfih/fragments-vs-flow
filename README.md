# Fragments-vs-Flow RAG Pipeline Project

This project contains the runnable thesis pipeline code as a standalone copy.
It includes:

- `app.py`
- `requirements.txt`
- `src/` runtime code
- `.env.example`
- `data/inputs/IN3240/` synthetic example inputs and test set

## Install

Use Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with the provider credentials you want to use.

Note: the `extract` step currently uses Docling for both slides and textbooks.
On a fresh machine, the first extraction run may download Docling model assets.

## Provider Configuration

Provider selection is environment-driven per role:

```text
GENERATION_PROVIDER=watsonx|azure_openai|openai
EVALUATION_PROVIDER=azure_openai|openai
EMBEDDING_PROVIDER=watsonx|azure_openai|openai
```

In practice, the current evaluation path requires an OpenAI-compatible judge
backend for RAGAS collection metrics. So use:

- `EVALUATION_PROVIDER=azure_openai`, or
- `EVALUATION_PROVIDER=openai`

`watsonx` remains supported for generation, and can still be used for
embeddings if you want, but it is not supported for evaluation in this
project.

Default behavior:

```text
GENERATION_PROVIDER=watsonx
EVALUATION_PROVIDER=azure_openai
EMBEDDING_PROVIDER=azure_openai
```

Role-specific model overrides:

```text
GENERATION_MODEL=
EVALUATION_MODEL=
EMBEDDING_MODEL=
```

If those are left empty, provider-specific fallbacks are used:

- `watsonx` generation falls back to `WATSONX_MODEL_ID`
- `watsonx` embeddings fall back to `WATSONX_EMBEDDING_MODEL_ID`
- `azure_openai` generation and evaluation fall back to `AZURE_OPENAI_DEPLOYMENT_NAME`
- `azure_openai` embeddings fall back to `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `openai` requires an explicit role model env for each selected role

Default provider mix example:

```text
GENERATION_PROVIDER=watsonx
EVALUATION_PROVIDER=azure_openai
EMBEDDING_PROVIDER=azure_openai
```

Mixed-provider example:

```text
GENERATION_PROVIDER=openai
GENERATION_MODEL=gpt-4.1-mini
EVALUATION_PROVIDER=azure_openai
EMBEDDING_PROVIDER=watsonx
EMBEDDING_MODEL=ibm/slate-30m-english-rtrvr
```

## Expected Data Layout

By default, the pipeline expects course inputs under:

```text
data/inputs/<COURSE>/
```

Each course folder should contain:

```text
slides/
textbooks/
test_set/
```

This project already includes one example course:

```text
data/inputs/IN3240/
```

The included PDFs and questions are synthetic demo material created for this
project. They are included only to demonstrate the runtime pipeline and do not
reuse the original course content.

The included test-set folder includes both:

```text
test_set.json
test_set_with_labels.json
```

Derived outputs such as `output/`, `milvus/`, and `results/` are written under
the selected course workspace.

### Data Needed For The Full Pipeline

To run the full standard pipeline for one source end to end, the course
workspace needs:

- source PDFs under `slides/` or `textbooks/`
- `test_set/test_set.json`
- `test_set/test_set_with_labels.json`

In practice, the stages depend on data like this:

- `extract`, `chunk`, and `index` need the source PDF files
- `retrieve` needs the source PDFs plus `test_set.json`
- `generate` needs a saved retrieval export
- `evaluate` needs a saved generation export plus `test_set_with_labels.json`
- `ir-metrics` needs retrieval or generation exports plus `test_set_with_labels.json`
- `pipeline` needs all of the above because it runs the whole sequence

For the bundled `IN3240` demo course, both `test_set.json` and
`test_set_with_labels.json` are already included.

## How To Use

Inspect the available commands:

```bash
python3 app.py --help
```

### Minimal Standard Pipeline

```bash
python3 app.py extract --course in3240 --source slides --overwrite
python3 app.py chunk --course in3240 --source slides --chunk-mode token-window
python3 app.py index --course in3240 --source slides
python3 app.py retrieve --course in3240 --source slides --top-k 10
```

These four steps work directly with the included synthetic example data.

### Generation And Evaluation

`generate` works after retrieval has been exported:

```bash
python3 app.py generate --course in3240 --source slides --top-k 10
```

`evaluate` and `ir-metrics` can then run against the included synthetic labeled
test set:

```bash
python3 app.py evaluate --course in3240 --source slides --top-k 10
python3 app.py ir-metrics --course in3240 --source slides --k-values 1 3 5 7 10
```

### One-Command Standard Run

The bundled demo course already includes the data needed for the full standard
pipeline, so you can run it in one command:

```bash
python3 app.py pipeline --course in3240 --source slides --overwrite --chunk-mode token-window --top-k 10 --k-values 1 3 5 7 10
```

### Derived-Corpus Example

Renarrated slides:

```bash
python3 app.py renarrative --course in3240 --k-values 1 3 5 7 10
```

Fragmented textbooks:

```bash
python3 app.py fragment-textbooks --course in3240 --k-values 1 3 5 7 10
```

### Evaluation Help

```bash
python3 app.py generate --help
python3 app.py evaluate --help
```

## Notes

This project is meant to expose the supported runtime pipeline through
`app.py` only.

The included example PDFs and questions are synthetic demo content, so this
project does not redistribute the original course materials.
