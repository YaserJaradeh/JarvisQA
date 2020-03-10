# JARVIS QA System
A prototype question answering systems on scholarly tabular data

## Important files
* The **datasets** folder contains the ORKG dataset and the TabMCQ dataset in the format that can be read by JarvisQA evaluation script.

* The **eval-results** folder contains the results of the TPDL2020 experimental evaluation.

### What does each file do?
To reproduce the TPDL2020 reported results you only need to run the `tpdl2020_eval.py` script.

### Easy setup
The system can be ran using docker

To build the docker image: `docker build . -t jarvis`

To run the docker image: `docker run --name jarvis jarvis`

Then just call the python file that you want executed e.g. `python file.py`

### Normal setup
You only need to have Python 3.6 and install all the requirement packages via:

`pip install -r requirements.txt`

and then just run the script that you need `python file.py`

#### Note
you need to have a Apache Solr instance running to evaluate the Lucene baseline. An easy method to run this using docker is
 
`docker run -d -p 8983:8983 --name my_solr solr solr-precreate gettingstarted`