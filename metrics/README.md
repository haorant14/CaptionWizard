# Metrics

There are two notebooks in this folder:
* `BERT_Embedding_Metrics.ipynb` - This notebook contains the code used to implement the BERT embedding feature to measure similarity between the embeddings of generated captions and 'likable' captions.
    * The notebook utilizes `../data/clean_data.csv` and `../data/ALL_PREDICTIONS_1200.csv`, as well as loading the BERT embeddings from `../data/clean_embeddings.npy`
* `BLEU_METEOR_Model_Evaluation.ipynb` - This notebook contains the code used to implement the BLEU and METEOR metrics to evaluate the performance of the model.
    * The notebook utilizes only `../data/ALL_PREDICTIONS_1200.csv`

For more information about the data, please see [here](../data/README.md)