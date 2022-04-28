# Reproduce results

## Pretraining
### Generate train metadata file
```
python mk_metadata.py -d <path to hotels50k> --hotels-50k --min-sample-limit 25 --max-sample-limit 75 -o data/pretrain_metadata.yml
```

## Generate train metadata file
<!-- ```
python mk_metadata.py -d data/hotel-id-to-combat-human-trafficking-2022-fgvc9 -o data/train_metadata.yml
``` -->

## Ideas
- `cameratransform` python package

### Ensemble using ranking
Have multiple similarity models, compute average highest rank to generate ranking.
