from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "fix_ud_childes_bg_checks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fix_ud_childes_bg_checks", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fix_sentence_merges_split_ellipsis():
    module = load_module()
    lines = [
        "# text = Мамо . . .",
        "1\tМамо\tмамо\tNOUN\t_\tCase=Voc\t0\troot\t_\t_",
        "2\t.\t.\tPUNCT\tpunct\t_\t1\tpunct\t_\t_",
        "3\t.\t.\tPUNCT\tpunct\t_\t2\tpunct\t_\t_",
        "4\t.\t.\tPUNCT\tpunct\t_\t3\tpunct\t_\t_",
    ]

    fixed_lines, counts = module.fix_sentence(lines)

    assert fixed_lines == [
        "# text = Мамо ...",
        "1\tМамо\tмама\tNOUN\t_\tDefinite=Ind|Gender=Fem|Number=Sing|Case=Voc\t0\troot\t_\t_",
        "2\t...\t...\tPUNCT\tpunct\t_\t1\tpunct\t_\t_",
    ]
    assert counts["ellipsis_merged"] == 1
    assert counts["lemma_fixed:мамо"] == 1
    assert counts["feats_fixed:мамо"] == 1


def test_fix_sentence_applies_requested_lexical_fixes():
    module = load_module()
    lines = [
        "# text = Айде , бе мамо , дай вкльучи втоето каъв xубаво .",
        "1\tАйде\tайде\tPART\tTm\t_\t6\tdiscourse\t_\t_",
        "2\t,\t,\tPUNCT\tpunct\t_\t3\tpunct\t_\t_",
        "3\tбе\tбе\tPART\t_\t_\t6\tdiscourse\t_\t_",
        "4\tде\tPART\tPART\tTe\tAspect=Imp|Mood=Ind|Number=Sing|Person=3\t6\tdiscourse\t_\t_",
        "5\tще\tща\tPART\tTx\tAspect=Imp|Mood=Ind|Number=Sing|Person=3\t6\taux\t_\t_",
        "6\tдай\tдавам\tVERB\tVpptz--2s\tAspect=Perf|Mood=Imp|Number=Sing|Person=2|VerbForm=Fin\t0\troot\t_\t_",
        "7\tмамо\tмамa\tNOUN\t_\tCase=Voc\t6\tvocative\t_\t_",
        "8\tвкльучи\tвкльуча\tVERB\tVpptz--2s\tAspect=Perf|Mood=Imp|Number=Sing|Person=2|VerbForm=Fin\t6\tccomp\t_\t_",
        "9\tвтоето\tмой\tPRON\t_\tDefinite=Def|Gender=Neut|Number=Sing\t10\tdet\t_\t_",
        "10\tкаъв\tКаъв\tDET\t_\tGender=Masc|Number=Sing|PronType=Int\t6\tobj\t_\t_",
        "11\txубаво\txубаво\tADV\tDm\tDegree=Pos\t6\tadvmod\t_\t_",
        "12\t.\t.\tPUNCT\tpunct\t_\t11\tpunct\t_\t_",
    ]

    fixed_lines, counts = module.fix_sentence(lines)
    fixed_tokens = [line.split("\t") for line in fixed_lines if not line.startswith("#")]

    assert fixed_lines[0] == "# text = Айде , бе мамо , дай включи твоето какъв xубаво ."
    assert fixed_tokens[0][3] == "INTJ"
    assert fixed_tokens[0][2] == "хайде"
    assert fixed_tokens[0][5] == "_"
    assert fixed_tokens[2][3] == "INTJ"
    assert fixed_tokens[2][2] == "бе"
    assert fixed_tokens[2][5] == "_"
    assert fixed_tokens[3][2] == "де"
    assert fixed_tokens[3][3] == "INTJ"
    assert fixed_tokens[3][5] == "_"
    assert fixed_tokens[4][2] == "ще"
    assert fixed_tokens[4][3] == "AUX"
    assert fixed_tokens[4][5] == "_"
    assert fixed_tokens[5][2] == "дам-(се)"
    assert fixed_tokens[6][2] == "мама"
    assert fixed_tokens[6][3] == "NOUN"
    assert fixed_tokens[6][5] == "Definite=Ind|Gender=Fem|Number=Sing|Case=Voc"
    assert fixed_tokens[7][1] == "включи"
    assert fixed_tokens[7][2] == "включа"
    assert fixed_tokens[8][1] == "твоето"
    assert fixed_tokens[8][2] == "твой"
    assert fixed_tokens[9][1] == "какъв"
    assert fixed_tokens[9][2] == "какъв"
    assert fixed_tokens[10][2] == "хубав"
    assert fixed_tokens[11][6] == "6"
    assert counts["upos_intj_fixed"] == 3
    assert counts["final_punct_attached_to_root"] == 1
    assert counts["surface_fixed:втоето"] == 1
    assert counts["surface_fixed:каъв"] == 1
    assert counts["lemma_fixed:какъв"] == 1
    assert counts["lemma_fixed:айде"] == 1
    assert counts["lemma_fixed:де"] == 1
    assert counts["lemma_fixed:ще"] == 1
    assert counts["feats_fixed:де"] == 1
    assert counts["feats_fixed:ще"] == 1
    assert counts["feats_fixed:мамо"] == 1


def test_fix_sentence_handles_babo_tate_and_vizh():
    module = load_module()
    lines = [
        "# text = Бабо , тате , виж !",
        "1\tБабо\tбабa\tNOUN\t_\tCase=Voc\t4\tvocative\t_\t_",
        "2\t,\t,\tPUNCT\tpunct\t_\t1\tpunct\t_\t_",
        "3\tтате\tтати\tNOUN\t_\tCase=Voc\t4\tvocative\t_\t_",
        "4\tвиж\tвидя-(се)\tVERB\tVpptz--2s\tAspect=Perf|Mood=Imp|Number=Sing|Person=2|VerbForm=Fin\t0\troot\t_\t_",
        "5\t!\t!\tPUNCT\tpunct\t_\t3\tpunct\t_\t_",
    ]

    fixed_lines, counts = module.fix_sentence(lines)
    fixed_tokens = [line.split("\t") for line in fixed_lines if not line.startswith("#")]

    assert fixed_tokens[0][2] == "баба"
    assert fixed_tokens[0][3] == "NOUN"
    assert fixed_tokens[0][5] == "Definite=Ind|Gender=Fem|Number=Sing|Case=Voc"
    assert fixed_tokens[2][2] == "тате"
    assert fixed_tokens[2][3] == "NOUN"
    assert fixed_tokens[2][5] == "Definite=Ind|Gender=Masc|Number=Sing"
    assert fixed_tokens[3][2] == "виждам-(се)"
    assert fixed_tokens[4][6] == "4"
    assert counts["lemma_fixed:бабо"] == 1
    assert counts["lemma_fixed:тате"] == 1
    assert counts["lemma_fixed:виж"] == 1
    assert counts["feats_fixed:бабо"] == 1
    assert counts["feats_fixed:тате"] == 1


def test_fix_sentence_normalizes_tati_to_tate_profile():
    module = load_module()
    lines = [
        "# text = Тати .",
        "1\tТати\tтати\tINTJ\tI\t_\t0\troot\t_\t_",
        "2\t.\t.\tPUNCT\tpunct\t_\t1\tpunct\t_\t_",
    ]

    fixed_lines, counts = module.fix_sentence(lines)
    fixed_tokens = [line.split('\t') for line in fixed_lines if not line.startswith('#')]

    assert fixed_tokens[0][1] == "Тати"
    assert fixed_tokens[0][2] == "тате"
    assert fixed_tokens[0][3] == "NOUN"
    assert fixed_tokens[0][5] == "Definite=Ind|Gender=Masc|Number=Sing"
    assert counts["lemma_fixed:тати"] == 1
    assert counts["upos_fixed:тати"] == 1
    assert counts["feats_fixed:тати"] == 1


def test_fix_sentence_preserves_lexical_shte_as_verb():
    module = load_module()
    lines = [
        "# text = Сашето не ще .",
        "1\tСашето\tсаше\tPROPN\t_\tDefinite=Def|Gender=Neut|Number=Sing\t3\tnsubj\t_\t_",
        "2\tне\tне\tPART\tTn\tPolarity=Neg\t3\tadvmod\t_\t_",
        "3\tще\tща\tVERB\tVpitf-r3s\tAspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\t_",
        "4\t.\t.\tPUNCT\tpunct\t_\t3\tpunct\t_\t_",
    ]

    fixed_lines, counts = module.fix_sentence(lines)
    fixed_tokens = [line.split('\t') for line in fixed_lines if not line.startswith('#')]

    assert fixed_tokens[2][1] == "ще"
    assert fixed_tokens[2][2] == "ща"
    assert fixed_tokens[2][3] == "VERB"
    assert fixed_tokens[2][5] == "Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act"
    assert "lemma_fixed:ще" not in counts
