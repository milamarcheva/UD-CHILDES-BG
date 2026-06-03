from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "extract_projective_cds.py"


def load_module():
    spec = importlib.util.spec_from_file_location("extract_projective_cds", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ensure_sentence_has_sent_id_adds_missing_id():
    module = load_module()
    lines = [
        "# text = Къпи кукличката .",
        "1\tКъпи\tкъпя-(се)\tVERB\tVpptz--2s\tAspect=Imp|Mood=Imp|Number=Sing|Person=2|VerbForm=Fin\t0\troot\t_\t_",
        "2\tкукличката\tкукличка\tNOUN\tNcfsd\tDefinite=Def|Gender=Fem|Number=Sing\t1\tobj\t_\t_",
    ]

    actual = module.ensure_sentence_has_sent_id(lines, 17)

    assert actual[0] == "# sent_id = 17"
    assert actual[1:] == lines


def test_ensure_sentence_has_sent_id_preserves_existing_id():
    module = load_module()
    lines = [
        "# sent_id = 91",
        "# text = Къпи кукличката .",
        "1\tКъпи\tкъпя-(се)\tVERB\tVpptz--2s\tAspect=Imp|Mood=Imp|Number=Sing|Person=2|VerbForm=Fin\t0\troot\t_\t_",
    ]

    actual = module.ensure_sentence_has_sent_id(lines, 17)

    assert actual == lines


def test_set_sentence_metadata_replaces_existing_id():
    module = load_module()
    lines = [
        "# sent_id = 91",
        "# text = Къпи кукличката .",
        "1\tКъпи\tкъпя-(се)\tVERB\tVpptz--2s\tAspect=Imp|Mood=Imp|Number=Sing|Person=2|VerbForm=Fin\t0\troot\t_\t_",
    ]

    actual = module.set_sentence_metadata(
        lines,
        {
            "sent_id": "LLM_000091",
            "child_age": "2;03.15",
            "participant_role": "ALE",
            "original_utterance": "obicham",
        },
    )

    assert actual[0] == "# sent_id = LLM_000091"
    assert actual[1] == "# child_age = 2;03.15"
    assert actual[2] == "# participant_role = ALE"
    assert actual[3] == "# original_utterance = obicham"
    assert actual[4:] == lines[1:]


def test_kept_sentences_from_ranges_inserts_missing_sent_ids(tmp_path: Path):
    module = load_module()
    doc_dir = tmp_path / "annotation" / "ALE_cds.conllu"
    cur_dir = tmp_path / "curation" / "ALE_cds.conllu"
    doc_dir.mkdir(parents=True)
    cur_dir.mkdir(parents=True)

    (cur_dir / "CURATION_USER.conllu").write_text(
        "\n".join(
            [
                "# text = Първо .",
                "1\tПърво\tпърво\tADV\t_\t_\t0\troot\t_\t_",
                "",
                "# text = Второ .",
                "1\tВторо\tвторо\tADV\t_\t_\t0\troot\t_\t_",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kept, total = module.kept_sentences_from_ranges(
        doc_dir,
        tmp_path / "curation",
        [(1, 2, "curated")],
        all_annotated=True,
    )

    assert total == 2
    assert kept == [
        "# sent_id = 1\n# text = Първо .\n1\tПърво\tпърво\tADV\t_\t_\t0\troot\t_\t_",
        "# sent_id = 2\n# text = Второ .\n1\tВторо\tвторо\tADV\t_\t_\t0\troot\t_\t_",
    ]


def test_aligned_sent_ids_from_csv_matches_source_order(tmp_path: Path):
    module = load_module()
    project_root = tmp_path / "project"
    source_dir = project_root / "source"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "ALE_cs.conllu"
    source_file.write_text(
        "\n".join(
            [
                "# text = Обичам .",
                "1\tОбичам\tобичам\tVERB\t_\t_\t0\troot\t_\t_",
                "",
                "# text = Не е .",
                "1\tНе\tне\tPART\t_\t_\t2\tadvmod\t_\t_",
                "2\tе\tсъм\tAUX\t_\t_\t0\troot\t_\t_",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "labling.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "Name",
                "Participant",
                "Age",
                "Utterance",
                "Manually_corrected",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "utterance_id": "LLM_000001",
                "Name": "ALE",
                "Participant": "ALE",
                "Age": "2;03.15",
                "Utterance": "obicham",
                "Manually_corrected": "обичам.",
            }
        )
        writer.writerow(
            {
                "utterance_id": "LLM_000002",
                "Name": "ALE",
                "Participant": "ALE",
                "Age": "2;03.15",
                "Utterance": "ne e",
                "Manually_corrected": "не е.",
            }
        )

    actual = module.aligned_sent_ids_from_csv(project_root, csv_path)

    assert actual == {"ALE_cs.conllu": ["LLM_000001", "LLM_000002"]}


def test_aligned_sentence_metadata_from_csv_matches_source_order(tmp_path: Path):
    module = load_module()
    project_root = tmp_path / "project"
    source_dir = project_root / "source"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "ALE_cs.conllu"
    source_file.write_text(
        "\n".join(
            [
                "# text = Обичам .",
                "1\tОбичам\tобичам\tVERB\t_\t_\t0\troot\t_\t_",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "labling.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "Name",
                "Participant",
                "Age",
                "Utterance",
                "Manually_corrected",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "utterance_id": "LLM_000001",
                "Name": "ALE",
                "Participant": "ALE",
                "Age": "2;03.15",
                "Utterance": "obicham",
                "Manually_corrected": "обичам.",
            }
        )

    actual = module.aligned_sentence_metadata_from_csv(project_root, csv_path)

    assert actual == {
        "ALE_cs.conllu": [
            {
                "sent_id": "LLM_000001",
                "child_age": "2;03.15",
                "participant_role": "ALE",
                "original_utterance": "obicham",
            }
        ]
    }


def test_kept_sentences_from_ranges_uses_aligned_utterance_ids(tmp_path: Path):
    module = load_module()
    doc_dir = tmp_path / "annotation" / "ALE_cds.conllu"
    cur_dir = tmp_path / "curation" / "ALE_cds.conllu"
    doc_dir.mkdir(parents=True)
    cur_dir.mkdir(parents=True)

    (cur_dir / "CURATION_USER.conllu").write_text(
        "\n".join(
            [
                "# sent_id = 1",
                "# text = Първо .",
                "1\tПърво\tпърво\tADV\t_\t_\t0\troot\t_\t_",
                "",
                "# sent_id = 2",
                "# text = Второ .",
                "1\tВторо\tвторо\tADV\t_\t_\t0\troot\t_\t_",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kept, total = module.kept_sentences_from_ranges(
        doc_dir,
        tmp_path / "curation",
        [(1, 2, "curated")],
        all_annotated=True,
        metadata_by_doc={
            "ALE_cds.conllu": [
                {
                    "sent_id": "LLM_000010",
                    "child_age": "2;03.15",
                    "participant_role": "MOTHER",
                    "original_utterance": "purvo",
                },
                {
                    "sent_id": "LLM_000020",
                    "child_age": "2;03.15",
                    "participant_role": "MOTHER",
                    "original_utterance": "vtoro",
                },
            ]
        },
    )

    assert total == 2
    assert kept == [
        "# sent_id = LLM_000010\n# child_age = 2;03.15\n# participant_role = MOTHER\n# original_utterance = purvo\n# text = Първо .\n1\tПърво\tпърво\tADV\t_\t_\t0\troot\t_\t_",
        "# sent_id = LLM_000020\n# child_age = 2;03.15\n# participant_role = MOTHER\n# original_utterance = vtoro\n# text = Второ .\n1\tВторо\tвторо\tADV\t_\t_\t0\troot\t_\t_",
    ]
