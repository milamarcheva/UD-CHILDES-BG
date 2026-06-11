# UD-CHILDES-BG
Bulgarian CHILDES Universal Dependency treebank and accompanying analysis.

The manually verified treebank data is in `UD-CHILDES-BG_cs_and_cds/`.
Utterance-level metadata keyed by `UTT_...` identifiers, including child name, age, and speaker role (`Participant`), is in `dfs/LabLing_stratified_with_manual_with_ids.csv`.

## Annotator agreement
Example usage of eval_agreement.py:  

python3 eval_agreement.py \
  --doc BOG_cs.conllu:mila,yoana:1-97 \
  --doc ELI_cs.conllu:mila,tsvetelina:1-90 \
  --project-root inception_annotations/bg_childes_tb_project_2026-03-05_1041

python3 eval_agreement.py \
  --doc ALE_cds.conllu:mila,yasena:1-90 \
  --doc BOG_cds.conllu:mila,yoana:1-148 \
  --project-root inception_annotations/bg_childes_tb_project_2026-03-05_1041

## Extract stitched manually annotated CoNLL-U
Use `extract_projective_cds.py` to combine manually annotated ranges from
multiple annotators into one CoNLL-U file per document.

- Default mode writes only projective sentences.
- `--all-annotated` writes all selected manually annotated sentences.
- `--v2` uses the expanded hard-coded range table for both `cs` and `cds`.

Example: write all manually annotated sections for the `2026-05-08` project to
`inception_annotations/bg_childes_tb_project_2026-05-08_2004/manually annotated/`

```bash
python3 extract_projective_cds.py \
  --project-root inception_annotations/bg_childes_tb_project_2026-05-08_2004 \
  --v2 \
  --all-annotated
```

Example: write only projective stitched sentences

```bash
python3 extract_projective_cds.py \
  --project-root inception_annotations/bg_childes_tb_project_2026-05-08_2004 \
  --v2
```

## Gold (manually corrected) vs parser (Stanza)
Use `eval_gold_vs_stanza.py` to compare stitched gold slices against automatic
Stanza parses stored in `source/`. Each `--gold` argument is:

`DOC:SRC:START-END`

where `SRC` is either an annotator filename stem (e.g. `mila`, `tsvetina`) or
`curation` for `curation/<DOC>/CURATION_USER.conllu`.

Example for the `2026-05-08` range split:

```bash
python3 eval_gold_vs_stanza.py \
  --project-root inception_annotations/bg_childes_tb_project_2026-05-08_2004 \
  --gold ALE_cs.conllu:yasena:1-65 \
  --gold ALE_cs.conllu:mila:66-245 \
  --gold ALE_cs.conllu:yasena:246-320 \
  --gold ALE_cs.conllu:mila:321-379 \
  --gold ALE_cds.conllu:curation:1-90 \
  --gold ALE_cds.conllu:mila:91-215 \
  --gold ALE_cds.conllu:yasena:216-320 \
  --gold ALE_cds.conllu:mila:321-430 \
  --gold BOG_cs.conllu:curation:1-97 \
  --gold BOG_cds.conllu:curation:1-148 \
  --gold ELI_cs.conllu:curation:1-88 \
  --gold ELI_cs.conllu:tsvetelina:89-121 \
  --gold ELI_cds.conllu:tsvetelina:1-160 \
  --gold ELI_cds.conllu:mila:161-385 \
  --gold ELI_cds.conllu:tsvetelina:386-560 \
  --gold ELI_cds.conllu:mila:561-706 \
  --gold SIM_cs.conllu:tsvetina:1-100 \
  --gold SIM_cs.conllu:mila:101-400 \
  --gold SIM_cs.conllu:tsvetina:401-504 \
  --gold SIM_cs.conllu:curation:505-515 \
  --gold SIM_cs.conllu:mila:516-924 \
  --gold SIM_cds.conllu:tsvetina:1-160 \
  --gold SIM_cds.conllu:curation:161-335 \
  --gold SIM_cds.conllu:mila:336-805 \
  --gold TEF_cs.conllu:ivelina:1-180 \
  --gold TEF_cs.conllu:tsvetelina:181-275 \
  --gold TEF_cs.conllu:yoana:276-336 \
  --gold TEF_cds.conllu:ivelina:1-265 \
  --gold TEF_cds.conllu:yoana:266-392
```

Add `--top-k-relations 20` to print more relations in the per-relation F1
table, or `--top-k-relations 0` to print all relations.

## Citations

- When using UD-CHILDES-BG please cite: 
```bibtex
@inproceedings{marcheva-nash-etal-2026-ud-childes-bg,
  author    = {Marcheva-Nash, Mila and Chantova, Yasena and Kirilova, Tsvetina and Pavlova, Ivelina and Stefanova, Tsvetelina and Vasileva, Yoana and Sun, Weiwei},
  title     = {{UD-CHILDES-BG}: A Dependency Treebank of Bulgarian Child and Child-Directed Speech},
  booktitle = {Proceedings of the 20th Linguistic Annotation Workshop},
  year      = {2026},
  address   = {San Diego, United States of America},
  note      = {To appear}
}
```
- When using PS-CHILDES-BG please cite
```bibtex
@inproceedings{marcheva-nash-sun-2026-ps-childes-bg,
  author    = {Marcheva-Nash, Mila and Sun, Weiwei},
  title     = {{PS-CHILDES-BG}: A Constituency Treebank of Bulgarian Morphemically Tokenised Child-Directed Speech},
  booktitle = {Proceedings of the Seventh International Conference on Computational Linguistics in Bulgaria},
  year      = {2026},
  address   = {Sofia, Bulgaria},
  publisher = {Department of Computational Linguistics, Institute for Bulgarian Language, Bulgarian Academy of Sciences},
  note      = {To appear}
}
```
- Please also cite the Bulgarian CHILDES dataset
```bibtex
@misc{PopovaPopov2020,
  doi = {10.21415/PHWH-J834},
  url = {https://childes.talkbank.org/access/Slavic/Bulgarian/LabLing.html},
  author = {Popova, Velka and Popov, Dimitar},
  title = {{CHILDES} {Bulgarian} {LabLing Corpus}},
  publisher = {TalkBank},
  year = {2020}
}
```
