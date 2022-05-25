# Reproduce results

## Pretraining
### Generate train metadata file
```
python mk_metadata.py -d <path to hotels50k> --hotels-50k --min-sample-limit 25 --max-sample-limit 75 -o data/pretrain_metadata.yml
```

<!-- ## Generate train metadata file -->
<!-- ```
python mk_metadata.py -d data/hotel-id-to-combat-human-trafficking-2022-fgvc9 -o data/train_metadata.yml
``` -->

# for report:
- mention arcface not working wo pretraining, possibly badly configure hparams (though same as winning solution previous year)
- compare 'CELoss + pretraining' to 'CELoss' normal
- compare 'CELoss + pretraining' to 'ArcFace + pretraining'
- compare augmentations.