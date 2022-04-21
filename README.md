# Reproduce results

## Generate train metadata file
```
python mk_metadata.py -d data/hotel-id-to-combat-human-trafficking-2022-fgvc9 -o data/train_metadata.yml
```

## Ideas
- `cameratransform` python package

### Ensemble using ranking
Have multiple similarity models, compute average highest rank to generate ranking.
