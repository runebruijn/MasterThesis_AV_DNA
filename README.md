# MasterThesis_AV_DNA
Master Thesis of Rune Bruijn about the influence of genetic and situational factors on writing style.

<h2>Description</h2>
During this Master Thesis of Rune Bruijn, the influence of genetic and situational factors on writing is researched, by using authorship verification models (GLAD, SVR, Bleached SVR, BERTje) in combination with texts from twins and siblings.

<h2>Models</h2>
This folder contains all the verification models: GLAD, SVR, Bleached SVR and BERTje.

<h3>GLAD</h3>
For information on the GLAD model, please see: https://github.com/pan-webis-de/glad
Running the model: python3 glad-main.py "../../Data/GLAD/[TRAINING FOLDER] ../../Data/GLAD/[TEST FOLDER]", where [TRAINING FOLDER] and [TEST FOLDER] are replaced with the path of the training folders and test folders containing the DU001, DU002, DU003, etc. folders.

<h3>Other Models</h3>
Running the models: "python3 [MODEL].py ../../Data/Other\ Models/[TRAINING FILE] ../../Data/Other\ Models/[TEST FILE]", where [TRAINING FILE] and [TEST FILE] are replaced with the path of the training file and test file.

<h2>Data</h2>
This folder contains all the data sets that are used in this research, where the structure of the GLAD data is different as the GLAD model reads the data in a different way.

<h2>Thesis Report</h2>
This folder contains the report (PDF) of the thesis.

<h2>Human Evaluation</h2>
This folder contains the texts and responses of the human evaluation on authorship verification that was performed during this research.
