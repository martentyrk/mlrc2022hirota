## Dependencies

### For LSTM classifier

```
- Python 3.7
- numpy 1.21.2 
- pytorch 1.9
- torchtext 0.10.0 
- spacy 3.4.0 
- sklearn 1.0.2 
- nltk 3.6.3
- captum 0.6.0
- transformers 4.26.0
- pandas 1.5.3
```

### For BERT classifier
```
- Python 3.7
- numpy 1.21.2 
- pytorch 1.4
- transformers 4.0.1
- spacy 2.3
- sklearn 1.0.2 
- nltk 3.6.3
- captum 0.6.0
- pandas 1.5.3
```

### Commands
Commands to run the code and more details about the original paper can 
be found on the GitHub page https://github.com/rebnej/lick-caption-bias

## Running the experiments:

### LIC scores for LSTM gender and race

In order to run the experiments to reproduce the LIC scores for LSTM race and gender the 
following steps must be taken:

Markup : 
1. Set up the LSTM environment with the dependencies shown above
2. In the terminal use the following line of code to run all LSTM models with 10 seeds:
```
sbatch scripts/run_all_lstm_models.job
```
The following snippet will call on run_lstm_gender.job to then run the lstm_leakage.py file and compute the scores. 
When looked inside the run_all_lstm_models.job file then there is a -J flag there after which comes a long name. This name represents the names of all output files for each model, so we recommend that to change to something descriptive in order to differentiate between files later.

#### Changing between LIC_m and LIC_d
The LIC_m and LIC_d generation can be changed in the ```scripts/run_lstm_gender.job``` file by changing ```--calc_model_leak True``` to ```--calc_ann_leak True```. 
```--calc_model_leak True``` will compute the LIC_m score and ```--calc_ann_leak True``` will compute the LIC_d score.

#### Changing between gender and race
In order to run experiments for race, then we have a separate .job file for that and it can be run by calling 
```
sbatch scripts/run_all_lstm_race_models.job
```





