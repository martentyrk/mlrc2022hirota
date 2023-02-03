# Table of Contents
1. [Dependencies](#dependecies)
2. [Reference to the original GitHub](#commands)
3. [Reproduction of the results](#reproduction-of-the-results)
    1. [LIC scores for the LSTM](#experiments_lstm)
    2. [LIC scores for the BERT](#experiments_bert)
    3. [Calculating accuracy metrics](#calculating-accuracies)
4. [Extension](#extension)
   1. [Integrated gradients](#integrated-gradients)
   2. [Data set dissection](#data-set-dissection)


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
- pycocoevalcap 1.2
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
- pycocoevalcap 1.2
- pandas 1.5.3
```

## Commands <a name="commands"></a>
Commands to run the code and more details about the original paper can 
be found on the GitHub page https://github.com/rebnej/lick-caption-bias

# Reproduction of the results: <a name="reproduction"></a>

## LIC scores for LSTM gender and race <a name="experiments_lstm"></a>

In order to run the experiments to reproduce the LIC scores for LSTM race and gender the 
 LSTM environment with the dependencies shown above must be set up.

\\
In the terminal use the following line of code to run all LSTM models with 10 seeds:
```
sbatch scripts/run_all_lstm_models.job
```
The following snippet will call on run_lstm_gender.job to then run the lstm_leakage.py file and compute the scores. 
When looked inside the run_all_lstm_models.job file then there is a -J flag there after which comes a long name. This name represents the names of all output files for each model, so we recommend that to change to something descriptive in order to differentiate between files later.

### Changing between LIC_m and LIC_d
The LIC_m and LIC_d generation can be changed in the ```scripts/run_lstm_gender.job``` file by changing ```--calc_model_leak True``` to ```--calc_ann_leak True```. 
```--calc_model_leak True``` will compute the LIC_m score and ```--calc_ann_leak True``` will compute the LIC_d score.

### Changing between gender and race
In order to run experiments for race, then we have a separate .job file for that and it can be run by calling 
```sbatch scripts/run_all_lstm_race_models.job```

## LIC scores for BERT gender and race <a name="experiments_bert"></a>
Before running the following lines, the BERT environment must be set up that is shown above.

In the terminal use the following line of code to run all LSTM models with 10 seeds:
```sbatch scripts/run_all_bert_models.job```
The following snippet will call on run_bert_gender.job to then run the bert_leakage.py file and compute the scores. 
When looked inside the run_all_bert_models.job file then there is a -J flag there after which comes a long name. This name represents the names of all output files for each model, so we recommend that to change to something descriptive in order to differentiate between files later.

### Changing between LIC_m and LIC_d
The LIC_m and LIC_d generation can be changed in the ```scripts/run_bert_gender.job``` file by changing ```--calc_model_leak True``` to ```--calc_ann_leak True```. 
```--calc_model_leak True``` will compute the LIC_m score and ```--calc_ann_leak True``` will compute the LIC_d score.

### Changing between gender and race
In order to run experiments for race, then we have a separate .job file for that and it can be run by calling 
```sbatch scripts/run_all_bert_race_models.job```

### Changing between the pre-trained model and the fine-tuned model
In order to switch from fine-tuned to pre-trained, then the following flags need to be 
added to either the ``` run_bert_gender.job ``` or ``` run_bert_race.job ```:
``` --freeze_bert True --num_epochs 20 --learning_rate 5e-5 ```. The following flags 
freeze the BERT parameters, changes the number of epochs to 20 and sets a new learning 
rate that was described in the original paper.

## Calculating accuracies
In order to calculate the 4 accuracy metrics for each captioning model, please refer to the file ``` model_accuracy ```. In order to compute the accuracies for all captioning models, the following script can be run in the terminal:
``` srun python3 model_accuracy.py --cap_model *enter model name* ```

The part between the two asterisk symbols can be then filled in with one of the following model names: ``` [sat, nic, nic_plus, nic_equalizer, oscar, fc, att2in, transformer, updn] ```

# Extension

The extension part can be split into two: integrated gradients and data set dissection.

## Integrated gradients
 
The files correlating to the integrated gradients method all contain the keyword "captum", referring to the library we are using to run the experiments.

### Files explained
``` lstm_captum.py ``` - The file contains everything needed to get the attribution scores for both female and men. By running the file, we calculate the attribution scores and save them into a pickle file in the form or a dictionary. This dictionary (as a pickle file) will be available in the folders attributions/LSTM.

``` LSTM_captum_notebook.ipynb ```- Contains everything needed to run the 
visual experiments for the integrated gradients for the LSTM model. Also has a part specifically 
meant for the dissection of the attribution files 
generated by ``` lstm_captum.py ```.  The notebook contains the integrated gradients visualization generation for both human and model generated captions and all the files necessary to run the notebook have already been added to the GitHub repository under bias_data_for_ig/LSTM. In order to get the exact same visualizations as are shown in the paper, we recommend runnning it in the Google Colab environment.

``` Bertft_captum_notebook.ipynb ``` - Contains everything needed to run the 
visual experiments for the integrated gradients for the BERT model. Since the 
model weights were too big to be uploaded into the repository, we created a new 
anonymous Google Drive, where we uploaded the weights (link to the drive: 
https://drive.google.com/drive/folders/1Qj0YtdNRQPVdENhSTDuiNwyADupkW5r8?
usp=sharing). In order to get the best visualizations that match the ones in 
our paper, we recommend running the notebook in Google Colab.

NB! The notebook files also contain guidelines and descriptions what they are for and what is needed to run them.

### Running lstm_captum.py
To run the ``` lstm_captum.py ``` we recommend using a script called ``` scripts/run_all_captum_lstm.py ``` by simply using the command:
``` sbatch scripts/run_all_captum_lstm.job ```

The command generates attribution files for the models NIC, NIC+ and NIC+Equalizer, since these are the models we also used to run the experiment with.

## Data set dissection

The data set dissection is mainly done in a single file, by looking at its outputs. The file we used for this is ``` lstm_remove_duplicates.py ```, which in the function 
``` def make_train_test_split() ``` has a couple of lines removing the overlapping captions between train and test. The captions get removed from the test set.

All the data to validate the numbers in the tables shown in the report will get printed during each run of the file. The file will showcase how many samples were removed and what are the sizes of the new test and train set.

The caption removal is done in this code snippet:
```
if args.remove_duplicates:
        df_traincaptions = pd.DataFrame(d_train, columns=['img_id', 'pred', 'bb_gender'])
        set1 = set(df_traincaptions['pred'])
        d_test2 = []
        for entry in d_test:
            if entry['pred'] not in set1:
                d_test2.append(entry)
        d_test = d_test2
```

The command to run all models:

``` sbatch scripts/run_all_lstm_duplicates.job ```

