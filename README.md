# Reproduce results

## Install dependencies using anaconda
```
conda env update -f environment.yml
```

## Pretraining

### Obtain Hotels 50k dataset
A fork of the original hotels 50k dataset can be found here: https://github.com/NilsHasNoGithub/Hotels-50K. This contains some changes, since the original code for downloading the data did not work for us.

### Generate train metadata file
```
python mk_metadata.py -d <path to hotels50k> --hotels-50k --min-sample-limit 25 --max-sample-limit 75 -o data/pretrain_metadata.yml
```

### Pretrain the model
```
python train.py -t data/pretrain_metadata.yml -e <configuration>
```
Several configurations used to obtain the results can be found in the `configurations` directory. The best performing configuration is `configurations/experiment/efficient_net_m_pretrain.yml`.

## Training/finetuning
To train a model, given a configuration:
```
python train.py -t train_metadata.yml -e <configuration>
```
Several configurations used to obtain the results can be found in the `configurations` directory. The best performing configuration is `configurations/experiment/efficient_net_m_finetune.yml`, keep in mind that pretraining is required for this, and the model needs to be stored in the file `models/effnet_pretrained.pt`. The `extract_pretrained_model.py` script can help with converting the model. 

## Inspecting training/validation metrics
Run in the project root:
```
mlflow ui
```
And open the provided link in a browser.


## Inference
Use the `predict.py` script for inference:
```
python predict.py -d data/hotel-id-to-combat-human-trafficking-2022-fgvc9/test_images -c configurations/inference/efficient_net_m_finetuned.yml -t train_metadata.yml -o <path/to/your/submission.csv>
```
The best configuration was `configurations/inference/efficient_net_m_finetuned.yml`. This requires that the best model checkpoint from the training stage is copied to the location specified in the configurations file.

The rotation correction configurations also require a rotation correction model. A pretrained model can be downloaded for this purpose:
```
wget https://nilsgolembiewski.nl/public_files/uploads/4O9WycdrtzunBpLTGJKViPh2Nve6I1/rotation_model.zip -O rotation_model.zip && \
    unzip rotation_model.zip && \
    rm rotation_model.zip
```

## Kaggle submissions
In order to make a submission in kaggle, the project, including all python dependencies, must be uploaded as a kaggle dataset, where a notebook can be created to use all files.