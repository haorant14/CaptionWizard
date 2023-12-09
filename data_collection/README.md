# Data Collection

To obtain the dataset, please request it from Seungbae Kim. You can do that on his [website](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset).

## Overview
Due to the sheer size of the dataset, this code was designed to be run on Google Colab. Please ensure that you modify the paths to the data files accordingly.

## Instructions
1. Run the `CaptionWizard_JSON_extraction.ipynb` notebook to extract the JSON data (post likes, comment count, caption, etc.) from the raw data. This will return a csv that will be used in the next step. (Originally, this output csv was split into two parts due to computational reasons, but this is no longer necessary.)
- This code will output fully_processed_dataset.csv, which will be used with the output of the next step.

2. Run the `CaptionWizard_Influencer_data_collection.ipynb` notebook to extract information related to the influencers (such as follower count), as well as whether or not their account made a post within the dataset.
- this code will output available_influencers.csv and not_available_influencers.csv, which will be merged with fully_processed_dataset.csv to create the dataset that will be preprocessed further. [See here](../models/CaptionWizard_EDA_and_Model_1.ipynb) for the merging portion.
- When merging the csv from the previous step, ensure that the name of the file is correct, and that you don't need to worry about running each "half" of the csv separately.

