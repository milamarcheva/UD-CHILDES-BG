# bg-childes-ud
Bulgarian CHILDES Universal Dependency treebank and accompanying analysis.

## Annotator agreement
Example usage of eval_agreement.py:  

python3 eval_agreement.py \
  --doc BOG_cs.conllu:mila,yoana:1-97 \
  --doc ELI_cs.conllu:mila,tsvetelina:1-90 --project-root inception_annotations/bg_childes_tb_project_2026-03-05_1041