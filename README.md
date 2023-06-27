# A Deep Network DeepOpacityNet for Detection of Cataracts from Color Fundus Photographs
This repository provides scripts, test image samples, and attention map samples for the paper titled 'A Deep Network DeepOpacityNet for Detection of Cataracts from Color Fundus Photographs'.

## Instructions to set up
### Prerequisites
Have python 3.8, tensorflow 2.3, keras 2.4 installed locally.

### Clone the repository
```
git clone https://github.com/ncbi/DeepOpacityNet.git
cd DeepOpacityNet
```

### Create a virtual environment
```
python3.8 -m venv deepopacitynet

source deepopacitynet/bin/activate 
```

### Preprocessing
We provided our custom code to preprocesss color fundus photos with sample examples in the "Preprocessing" folder.

### The trained models
You can access the trained DeepOpacityNet model from the 'Model' folder. This model was trained on our development (i.e., internal) dataset. Also, we provided the codes used to generate the models used in the study in the "Model Codes" folder.

### Sample images
We included the subset of the test set as sample preprocessed images. This subset was used for the subjective grading in our study.


### Run the script
```
python classify_data.py --model_folder=Model --image_folder=CFP --output_file=predictions.csv
```
Please note that models and images are provided in the repository


### Check the output file
Compare predictions.csv with deepopacitynet_predictions.csv. If they are the same, it is ready for testing other images.

## Heatmpas of sample images
The samples can be accessed from 'CFP' folder.

## NCBI's Disclaimer
This tool shows the results of research conducted in the [Computational Biology Branch](https://www.ncbi.nlm.nih.gov/research/), [NCBI](https://www.ncbi.nlm.nih.gov/home/about). 

The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. 

More information about [NCBI's disclaimer policy](https://www.ncbi.nlm.nih.gov/home/about/policies.shtml) is available.

About [text mining group](https://www.ncbi.nlm.nih.gov/research/bionlp/).

## For Research Use Only
The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only. 

## Acknowledgement
This research was supported in part by the Intramural Research Program of the National Eye Institute, National Institutes of Health, Department of Health and Human Services, Bethesda, Maryland, and the National Center for Biotechnology Information, National Library of Medicine, National Institutes of Health. The sponsor and funding organization participated in the design and conduct of the study; data collection, management, analysis, and interpretation; and the preparation, review and approval of the manuscript.
The views expressed herein are those of the authors and do not reflect the official policy or position of Walter Reed National Military Medical Center, Madigan Army Medical Center, Joint Base Andrews, the U.S. Army Medical Department, the U.S. Army Office of the Surgeon General, the Department of the Air Force, the Department of the Army/Navy/Air Force, Department of Defense, the Uniformed Services University of the Health Sciences or any other agency of the U.S. Government. Mention of trade names, commercial products, or organizations does not imply endorsement by the U.S. Government.


## Cite our work
Elsawy, A*, Keenan, T.D*., Chen, Q*., ..., Chew, E.Y†, and Lu, Z†. 2023. [A Deep Network DeepOpacityNet for Detection of Cataracts from Color Fundus Photographs](https://www.nature.com/commsmed/). Commuincations Medicine.