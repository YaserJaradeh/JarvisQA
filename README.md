# JARVIS QA System
A prototype question answering systems on scholarly tabular data

### Important files
* The **datasets** folder contains the ORKG dataset and the TabMCQ dataset in the format that can be read by JarvisQA evaluation script.

* The **eval-results** folder contains the results of the TPDL2020 experimental evaluation.

### Docker
The system can be ran using docker

To build the docker image: `docker build . -t jarvis`

To run the docker image: `docker run --name jarvis jarvis`

Then just call the python file that you want executed e.g. `python file.py`
