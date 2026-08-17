from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Replacement:
    before: str
    after: str
    expected_count: int = 1


_EXPECTED_INPUT_SHA256 = {
    "tkg_qa_models/train_qa_model.py": (
        "43435cbd035d30121c8c6e230b200eae91a5e5879794619c97f467c5f78e8780"
    ),
    "tkg_qa_models/qa_baselines.py": (
        "6536e6584164c04dd4c3523b1ea3e93f6238b51a2cc3b092e96319c4fdbf398b"
    ),
    "tkg_qa_models/qa_datasets.py": (
        "5d745078b16a3197d77809c20c59cf893563919232be1e0a3611f5ba756a43df"
    ),
    "tkg_qa_models/utils.py": (
        "a46395821f35ad211e8f316c89a338d7599572f2d35168be2465c824309f8077"
    ),
    "tkg_qa_models/hard_supervision_functions.py": (
        "efe5b74fdd6512c4ca7ff1500984028330c5a35db481ab4c7c931787a2958adb"
    ),
}

_TRAIN_IMPORTS_BEFORE = """import os
import argparse
import torch
"""
_TRAIN_IMPORTS_AFTER = """import os
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
"""

_TRAIN_ARGS_BEFORE = """parser.add_argument('--temperature',     default=1.0,   type=str)

args = parser.parse_args()
print_info(args)

data_dir = '/Data/data'
"""
_TRAIN_ARGS_AFTER = """parser.add_argument('--temperature',     default=1.0,   type=str)
parser.add_argument('--seed',            default=1729, type=int)
parser.add_argument('--num_hops',        default=4, type=int)
parser.add_argument('--num_workers',     default=4, type=int)
parser.add_argument('--deterministic',   action='store_true')
parser.add_argument(
    '--temporal_hint_mode', choices=('hard', 'disabled'), default='hard'
)
parser.add_argument('--variant', choices=('standard', 'hard'), default='hard')
parser.add_argument('--predictions_file', default='', type=str)
parser.add_argument('--run_id', default='', type=str)

args = parser.parse_args()
print_info(args)

if args.deterministic:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.deterministic:
    torch.use_deterministic_algorithms(True)

data_dir = os.environ.get('SABET_DATA_DIR', '/Data/data')
output_dir = os.environ.get('SABET_OUTPUT_DIR', data_dir)


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_generator(split):
    split_offset = sum(ord(char) for char in split)
    generator = torch.Generator()
    generator.manual_seed(args.seed + split_offset)
    return generator


def _canonical_json(value):
    if isinstance(value, dict):
        return {str(key): _canonical_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_json(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _entity_label(dataset, raw_id):
    if args.dataset_name == 'MultiTQ':
        return str(raw_id)
    value = dataset.all_dicts['wd_id_to_text'].get(raw_id)
    label = '' if value is None else str(value).strip()
    return label or str(raw_id)


def _decode_ranked_index(dataset, answer_index):
    num_entities = len(dataset.all_dicts['ent2id'])
    if answer_index < num_entities:
        raw_id = dataset.all_dicts['id2ent'][answer_index]
        label = _entity_label(dataset, raw_id)
        return f'entity:{raw_id}', label
    raw_time = dataset.all_dicts['id2ts'][answer_index - num_entities]
    label = str(raw_time[0] if isinstance(raw_time, tuple) else raw_time)
    return f'time:{label}', label


def _decode_gold_answer(dataset, answer, declared_answer_type):
    try:
        is_entity = answer in dataset.all_dicts['ent2id']
    except TypeError:
        is_entity = False
    if not is_entity and isinstance(answer, int):
        is_entity = False
    elif not is_entity and isinstance(answer, tuple):
        is_entity = False
    elif not is_entity:
        is_entity = declared_answer_type == 'entity'
    if is_entity:
        label = _entity_label(dataset, answer)
        return f'entity:{answer}', label
    label = str(answer[0] if isinstance(answer, tuple) else answer)
    return f'time:{label}', label


def _question_text(question):
    paraphrases = question.get('paraphrases')
    if paraphrases:
        return str(paraphrases[0])
    return str(question.get('question', ''))


def _write_predictions(dataset, topk_ids, topk_labels, topk_scores, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f'{args.dataset_name}-{args.model}-{args.variant}-seed{args.seed}'
    with output_path.open('w', encoding='utf-8', newline='\\n') as handle:
        for index, question in enumerate(dataset.data):
            answer_type = str(question['answer_type'])
            decoded_gold = [
                _decode_gold_answer(dataset, answer, answer_type)
                for answer in sorted(question['answers'], key=str)
            ]
            question_type = question.get('qtype', question.get('type', ''))
            if isinstance(question_type, list):
                question_type = question_type[0]
            canonical_source = json.dumps(
                _canonical_json(question), sort_keys=True, ensure_ascii=False,
                separators=(',', ':'),
            ).encode('utf-8')
            row = {
                'schema_version': '1.1',
                'run_id': run_id,
                'dataset': args.dataset_name,
                'split': 'test',
                'model': args.model,
                'variant': args.variant,
                'seed': args.seed,
                'source_index': index,
                'qid': str(question.get('quid', question.get('uniq_id', index))),
                'question': _question_text(question),
                'question_type': str(question_type),
                'answer_type': answer_type,
                'gold_answer_ids': [item[0] for item in decoded_gold],
                'gold_answer_labels': [item[1] for item in decoded_gold],
                'predicted_answer_ids': topk_ids[index],
                'predicted_answer_labels': topk_labels[index],
                'predicted_scores': topk_scores[index],
                'source_record_sha256': hashlib.sha256(canonical_source).hexdigest(),
            }
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\\n')
    print(f'Saved {len(dataset.data)} prediction records to {output_path}')
"""

_EVAL_STATE_BEFORE = """    topk_answers = []
    total_loss   = 0
"""
_EVAL_STATE_AFTER = """    topk_answers = []
    topk_answer_ids = []
    topk_answer_labels = []
    topk_answer_scores = []
    total_loss   = 0
"""

_EVAL_PREDICTIONS_BEFORE = """            for s in scores:
                topk_answers.append(dataset.getAnswersFromScores(s, k=max_k))

            total_loss += qa_model.loss(scores, answers_khot.cuda()).item()
"""
_EVAL_PREDICTIONS_AFTER = """            for s in scores:
                topk_answers.append(dataset.getAnswersFromScores(s, k=max_k))
                values, indices = torch.topk(s, max_k, largest=True)
                decoded = [_decode_ranked_index(dataset, int(index)) for index in indices]
                topk_answer_ids.append([item[0] for item in decoded])
                topk_answer_labels.append([item[1] for item in decoded])
                topk_answer_scores.append([float(value) for value in values.detach().cpu()])

            total_loss += qa_model.loss(scores, answers_khot.cuda()).item()
"""

_EVAL_RETURN_BEFORE = """    for s in eval_log:
        print(s)
    return eval_accuracy_for_report, eval_log

def eval1("""
_EVAL_RETURN_AFTER = """    if predictions_file:
        _write_predictions(
            dataset,
            topk_answer_ids,
            topk_answer_labels,
            topk_answer_scores,
            predictions_file,
        )
    for s in eval_log:
        print(s)
    return eval_accuracy_for_report, eval_log

def eval1("""

_FINAL_EVAL_BEFORE = """score, log = eval(
    qa_model, test_dataset,
    batch_size=args.valid_batch_size,
    split='test',
    k=args.eval_k,
)
"""
_FINAL_EVAL_AFTER = """score, log = eval(
    qa_model, test_dataset,
    batch_size=args.valid_batch_size,
    split='test',
    k=args.eval_k,
    predictions_file=args.predictions_file or None,
)
"""

_EVAL_MODE_BEFORE = """    score, log = eval(
        qa_model, test_dataset,
        batch_size=args.valid_batch_size,
        split=args.eval_split, k=args.eval_k,
        hop_weights_file=hop_file
    )
"""
_EVAL_MODE_AFTER = """    score, log = eval(
        qa_model, test_dataset,
        batch_size=args.valid_batch_size,
        split=args.eval_split, k=args.eval_k,
        hop_weights_file=hop_file,
        predictions_file=args.predictions_file or None,
    )
"""

_PATCHES = {
    "tkg_qa_models/train_qa_model.py": (
        Replacement(_TRAIN_IMPORTS_BEFORE, _TRAIN_IMPORTS_AFTER),
        Replacement(_TRAIN_ARGS_BEFORE, _TRAIN_ARGS_AFTER),
        Replacement(
            "def eval(qa_model, dataset, batch_size=128, split='valid', k=200, "
            "hop_weights_file=None):",
            "def eval(qa_model, dataset, batch_size=128, split='valid', k=200, "
            "hop_weights_file=None, predictions_file=None):",
        ),
        Replacement("    num_workers   = 4", "    num_workers   = args.num_workers", 2),
        Replacement("    num_workers = 5", "    num_workers = args.num_workers"),
        Replacement(
            "    data_loader = DataLoader(\n"
            "        dataset, batch_size=batch_size, shuffle=False,\n"
            "        num_workers=num_workers, collate_fn=dataset._collate_fn,\n"
            "    )",
            "    data_loader = DataLoader(\n"
            "        dataset, batch_size=batch_size, shuffle=False,\n"
            "        num_workers=num_workers, collate_fn=dataset._collate_fn,\n"
            "        worker_init_fn=_seed_worker, generator=_loader_generator(split),\n"
            "    )",
        ),
        Replacement(
            "    data_loader   = DataLoader(\n"
            "        dataset, batch_size=batch_size, shuffle=False,\n"
            "        num_workers=num_workers, collate_fn=dataset._collate_fn,\n"
            "    )",
            "    data_loader   = DataLoader(\n"
            "        dataset, batch_size=batch_size, shuffle=False,\n"
            "        num_workers=num_workers, collate_fn=dataset._collate_fn,\n"
            "        worker_init_fn=_seed_worker, generator=_loader_generator(split),\n"
            "    )",
        ),
        Replacement(
            "    data_loader   = DataLoader(\n"
            "        dataset, batch_size=batch_size, shuffle=True,\n"
            "        num_workers=num_workers, collate_fn=dataset._collate_fn,\n"
            "    )",
            "    data_loader   = DataLoader(\n"
            "        dataset, batch_size=batch_size, shuffle=True,\n"
            "        num_workers=num_workers, collate_fn=dataset._collate_fn,\n"
            "        worker_init_fn=_seed_worker, generator=_loader_generator('train'),\n"
            "    )",
        ),
        Replacement(_EVAL_STATE_BEFORE, _EVAL_STATE_AFTER),
        Replacement(_EVAL_PREDICTIONS_BEFORE, _EVAL_PREDICTIONS_AFTER),
        Replacement(_EVAL_RETURN_BEFORE, _EVAL_RETURN_AFTER),
        Replacement(_EVAL_MODE_BEFORE, _EVAL_MODE_AFTER),
        Replacement(_FINAL_EVAL_BEFORE, _FINAL_EVAL_AFTER),
        Replacement(
            'f"results/{args.dataset_name}/{args.save_to}.log"',
            'f"{output_dir}/results/{args.dataset_name}/{args.save_to}.log"',
            3,
        ),
        Replacement(
            'f"results/{args.dataset_name}/{args.save_to}_test_hop_weights.pt"',
            'f"{output_dir}/results/{args.dataset_name}/'
            '{args.save_to}_test_hop_weights.pt"',
        ),
        Replacement(
            'f"{data_dir}"\n'
            '        f"/qa_models/{args.dataset_name}/{args.save_to}.ckpt"',
            'f"{output_dir}"\n'
            '        f"/qa_models/{args.dataset_name}/{args.save_to}.ckpt"',
        ),
        Replacement(
            'f"{data_dir}"\n'
            '    f"/qa_models/{args.dataset_name}/{args.save_to}.ckpt"',
            'f"{output_dir}"\n'
            '    f"/qa_models/{args.dataset_name}/{args.save_to}.ckpt"',
        ),
        Replacement(
            'f"{data_dir}"\n'
            '        f"/qa_models/{args.dataset_name}/{args.load_from}.ckpt"',
            'f"{output_dir}"\n'
            '        f"/qa_models/{args.dataset_name}/{args.load_from}.ckpt"',
        ),
        Replacement(
            'f"{data_dir}/qa_models/{args.dataset_name}"',
            'f"{output_dir}/qa_models/{args.dataset_name}"',
        ),
    ),
    "tkg_qa_models/qa_baselines.py": (
        Replacement("dtype=np.long", "dtype=np.int64"),
        Replacement(
            "        self.args = args\n\n        def pick_heads(dim):",
            "        self.args = args\n"
            "        self.temporal_hint_mode = getattr(args, 'temporal_hint_mode', 'hard')\n\n"
            "        def pick_heads(dim):",
        ),
        Replacement(
            "        t1_emb = self.tkbc_model.embeddings[2](t1)\n"
            "        t2_emb = self.tkbc_model.embeddings[2](t2)\n\n"
            "        # ── Encode question ",
            "        if self.temporal_hint_mode == 'hard':\n"
            "            t1_emb = self.tkbc_model.embeddings[2](t1)\n"
            "            t2_emb = self.tkbc_model.embeddings[2](t2)\n"
            "        else:\n"
            "            t1_emb = None\n"
            "            t2_emb = None\n\n"
            "        # ── Encode question ",
        ),
    ),
    "tkg_qa_models/qa_datasets.py": (
        Replacement(
            'data_dir = "/Data/data"',
            "data_dir = os.environ.get('SABET_DATA_DIR', '/Data/data')",
        ),
    ),
    "tkg_qa_models/utils.py": (
        Replacement("import pickle\n", "import os\nimport pickle\n"),
        Replacement(
            'data_root = "/Data/data"',
            "data_root = os.environ.get('SABET_DATA_DIR', '/Data/data')",
        ),
        Replacement(
            "base_path = f'{data_root}/data/{dataset_name}/kg/tkbc_processed_data'.format("
            "dataset_name=dataset_name)",
            "base_path = f'{data_root}/data/{dataset_name}/kg/tkbc_processed_data/"
            "{dataset_name}'",
        ),
    ),
    "tkg_qa_models/hard_supervision_functions.py": (
        Replacement("import random\n", "import os\nimport random\n"),
        Replacement(
            'data_root = "/Data/data"',
            "data_root = os.environ.get('SABET_DATA_DIR', '/Data/data')",
        ),
    ),
}


def prepare_instrumented_copy(*, source_root: Path, output_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    _verify_inputs(source_root)
    shutil.copytree(source_root, output_root, ignore=shutil.ignore_patterns(".git"))

    changed: dict[str, dict[str, str]] = {}
    for relative_path, replacements in _PATCHES.items():
        path = output_root / relative_path
        text = path.read_text(encoding="utf-8")
        before_sha256 = _sha256(path)
        for replacement in replacements:
            actual_count = text.count(replacement.before)
            if actual_count != replacement.expected_count:
                raise ValueError(
                    f"Patch anchor count for {relative_path} is {actual_count}; "
                    f"expected {replacement.expected_count}"
                )
            text = text.replace(
                replacement.before,
                replacement.after,
                replacement.expected_count,
            )
        path.write_text(text, encoding="utf-8", newline="\n")
        changed[relative_path] = {
            "input_sha256": before_sha256,
            "output_sha256": _sha256(path),
        }

    manifest = {
        "schema_version": "1.0",
        "patch_name": "sabet-tkgqa-reproduction-instrumentation-v5",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "changed_files": changed,
        "semantic_changes": [
            "configurable data root",
            "isolated configurable output root",
            "explicit deterministic seeds and DataLoader worker seeding",
            "NumPy compatibility alias replacement",
            "MultiTQ dictionary path compatibility with the released archive layout",
            "optional hard temporal-hint branch for reconstructed non-hard runs",
            "lossless per-question top-100 prediction export",
            "per-gold-answer namespace resolution for released mixed entity/time answer sets",
            "stable entity-ID fallback when a released English label is empty or missing",
        ],
        "non_goals": [
            "no scoring-equation change in hard-hint runs",
            "no optimizer, schedule, loss, or checkpoint-selection change",
            "no dataset record or native gold-membership change",
        ],
    }
    (output_root / "CODEX_REPRODUCTION_PATCH_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a hash-checked SABET-QA runnable copy")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_instrumented_copy(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _verify_inputs(source_root: Path) -> None:
    for relative_path, expected in _EXPECTED_INPUT_SHA256.items():
        path = source_root / relative_path
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Pinned source mismatch for {relative_path}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
