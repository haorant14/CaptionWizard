# Data Collection

To obtain the dataset, please request it from Seungbae Kim. You can do that on his [website](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset).

## Overview
Due to the sheer size of the dataset, this code was designed to be run on Google Colab. Please ensure that you modify the paths to the data files accordingly.

## Instructions
1. Run the `CaptionWizard_JSON_extraction.ipynb` notebook to extract the JSON data (post likes, comment count, caption, etc.) from the raw data. This will return a csv that will be used in the next step. (Originally, this output csv was split into two parts due to computational reasons, but this is no longer necessary.)

2. 
- When merging the csv from the previous step, ensure that the name of the file is correct, and that you don't need to worry about running each "half" of the csv separately.

