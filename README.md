# Table of Contents
1. [Dependencies](#dependecies)
2. [Reference to the original GitHub](#commands)
3. [Running the experiments](#experiments)
    1. [LIC scores for the LSTM](experiments_lstm)
    2. [LIC scores for the BERT](experiments_bert)



## Dependencies <a name="dependencies"></a>

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

## Commands <a name="commands"></a>
Commands to run the code and more details about the original paper can 
be found on the GitHub page https://github.com/rebnej/lick-caption-bias

## Running the experiments: <a name="experiments"></a>

### LIC scores for LSTM gender and race <a name="experiments_lstm"></a>

In order to run the experiments to reproduce the LIC scores for LSTM race and gender the 
 LSTM environment with the dependencies shown above must be set up.

\\
In the terminal use the following line of code to run all LSTM models with 10 seeds:
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

### LIC scores for BERT gender and race <a name="experiments_bert"></a>
Before running the following lines, the BERT environment must be set up that is shown above.

In the terminal use the following line of code to run all LSTM models with 10 seeds:
```
sbatch scripts/run_all_bert_models.job
```
The following snippet will call on run_bert_gender.job to then run the bert_leakage.py file and compute the scores. 
When looked inside the run_all_bert_models.job file then there is a -J flag there after which comes a long name. This name represents the names of all output files for each model, so we recommend that to change to something descriptive in order to differentiate between files later.

#### Changing between LIC_m and LIC_d
The LIC_m and LIC_d generation can be changed in the ```scripts/run_bert_gender.job``` file by changing ```--calc_model_leak True``` to ```--calc_ann_leak True```. 
```--calc_model_leak True``` will compute the LIC_m score and ```--calc_ann_leak True``` will compute the LIC_d score.

#### Changing between gender and race
In order to run experiments for race, then we have a separate .job file for that and it can be run by calling 
```
sbatch scripts/run_all_bert_race_models.job
```

#### Changing between the pre-trained model and the fine-tuned model
In order to switch from fine-tuned to pre-trained, then the following flags need to be 
added to either the ``` run_bert_gender.job ``` or ``` run_bert_race.job ```:
``` --freeze_bert True --num_epochs 20 --learning_rate 5e-5 ```. The following flags 
freeze the BERT parameters, changes the number of epochs to 20 and sets a new learning 
rate that was described in the original paper.







