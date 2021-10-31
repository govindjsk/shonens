## Description
This code contains following models and necessary trainer code to train these
models for the task of hyperlink prediction. Available models:
* SHONeN
* HGNNHyperlinkPrediction
* Node2VecHyperlinkPrediction
* HyperSAGNN

HGNN, Node2Vec and HyperSAGNN models take following arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset. Possible choices: email-enron,
                        contact-primary-school, NDC, DBLP, math-sx, contact-
                        high-school, MAG-Geo
  --ratio RATIO         Number of negative samples for each positive sample in
                        test set
  --model MODEL         Model to train: HGNN. HyperSAGNN, Node2Vec
  --max-epoch MAX_EPOCH
                        Number of training epochs


SHONeN model takes following arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset. Possible choices: email-enron,
                        contact-primary-school, NDC, DBLP, math-sx, contact-
                        high-school, MAG-Geo
  --ratio RATIO         Number of negative samples for each positive sample in
                        test set
  --max-epoch MAX_EPOCH
                        Total number of epochs to train for.
  --max-subset-size MAX_SUBSET_SIZE
                        Maximum size of subsets to be considered for
                        sub-hyperdges.


### Dependencies
Install follwing python libraries to be able to run this code on your system:
* networkx
* numpy
* scipy
* pytorch
* matplotlib
* scikit-learn
* node2vec (https://github.com/eliorc/node2vec)
* tqdm
* tensorboard
Run `pip install -r requirements.txt` to install all necessary dependencies.


## Executing the code
Run following bash scripts to obtain results from different models:
* hgnn.sh -- HGNNHyperlinkPrediction
* n2v.sh -- Node2VecHyperlinkPrediction
* hypersagnn.sh -- HyperSAGNN
* shonen.sh -- SHONeN

### Outputs
Output from the models will be stored in
`reports/{dataset}/{model_timstamp}.json` file. The output has following format:
```
for each epoch, a dictionary of following format is contained: {
    'train_repport': {
        'ROC': {value}
        'Accuracy': {value},
        'Precision': {value},
        'Recall': {value},
        'F1': {value},
    },

    'test_report': {
        'ROC': {value}
        'Accuracy': {value},
        'Precision': {value},
        'Recall': {value},
        'F1': {value},
    }
}
```
Moreover, `.logs` file contains logs which can be used to visualize through
tensorboard. Trained models are stored in
`reports/{dataset}/{model_timstamp}.pt` file.
Run, `tensorboard --logdir ./.logs` and open `localhost:6006` to visualize
the results.
