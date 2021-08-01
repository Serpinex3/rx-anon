# rx-anon
Repository containing code and experiments to anonymize heterogeneous data.

## Purpose
This project is part of my Master's Thesis on "A novel Approach of De-Identification of Heterogeneous Data based on a modified Mondrian Algorithm" at the University of Ulm in cooperation with BT.

## General Architecture
![Overview](https://drive.google.com/uc?export=view&id=1yx8mN1001bUKOqmm309onBZ0Gt4YNLZo)

## Preparation and Installation
In order to use the tool, you either need to download the latest docker image or install python dependencies yourself. Following are the instructions for both approaches.

### Using docker image

#### Requirements
- You need to have docker installed. Instruction on how to install the docker engine can be found [here](https://docs.docker.com/engine/install/).
- Moreover, for ease of use, we set up the instructions such that you can start the container using docker-compose. Therefore, you should have the docker-compose binary installed. On Windows, docker-compose comes with the docker engine. Instructions on how to install docker-compose on Linux can be found [here](https://docs.docker.com/compose/install/).

#### Instructions
1. Clone the repository or download it.
2. Change into the root directory of the project (`cd 2020ss-thesis-fabian`).
3. Pull the latest image of the tool by `docker-compose pull`. You can also always run this command to get the latest image for the app.
4. Run the app using docker-compose. To do so start the app using `docker-compose run --rm app`.

**Remark:** If you can't pull the image, try to login to docker hub. Use docker CLI to log into the repository by entering `docker login` and adding your credentials.

### Using Python packages

#### Requirements
You need to have python3.8 and pip installed. Preferably, you should work within a virtual environment to ensure not to influence your system dependencies.

#### Instructions
1. Clone the repository or download it.
2. Change into the root directory of the project (`cd 2020ss-thesis-fabian`).
3. Install and update the required packages using pip by using `pip install -U -r requirements/base.txt`. This might take a while since SpaCy models are downloaded and build from scratch.

## Usage
Whether you are within the docker container or you installed the packages manually, the usage of the tool remains the same.

The tool requires three input parameters, namely the input csv file which should be anonymized, a corresponding configuration describing important settings to apply, and the output file. An example would be:

```shell
python anon/main.py -i data/datasets/paper_example.csv -c data/configurations/blog_authorship_corpus.yaml -o data/results/paper_example_anonymized.csv
```

Moreover, you can enable verbose logging by adding the `-v` flag. Finally, if you want to anonymize one file in various ways (say run an experiment with different values of k) you might want to add the `-s` flag to use cached documents. This makes the processing way faster since for all textual documents the tool tries to use cached results from previous runs.

### Configuration
The tool allows for flexible configuration of the anonymization parameters.

The configuration consists of multiple sections. First, the anonymization parameters for the algorithm can be configured. Within the parameter section, the anonymization parameter k can be set to any integer number. Moreover, the partitioning strategy can be either **gdf** or **mondrian**. If you choose to use Mondrian partitioning, you can also specify a relational_weight parameter which determines the importance of relational attributes during the partitioning phase.
```yaml
parameters:
  k: 2
  strategy: mondrian
  relational_weight: 0.1
```

Next a section on natural language processing describes which model to use for analyzing texts. Currently supported models are **en_core_web_sm**, **en_core_web_md**, **en_core_web_lg**, and **en_core_web_trf**.
```yaml
nlp:
  model: en_core_web_trf
```

The following section on attributes describes attributes appearing in the dataset to be anonymized. In order to configure attributes, name them within the attributes section. For each attrybute, you can describe its type (either **nominal**, **ordinal**, **numerical** **date**, or **text**). Date attributes require also to have a format field which describes the date format. Additionally, numerical attributes support hierarchies, which will be used to recode numerical ranges using the nodes of the hierarchy. Moreover, the role of this attribute within the anonymization process can be described with the anonymization_type. Roles can be **direct_identifier**, **quasi_identifier**, or **text** for textual attributes. Moreover, our Mondrian implementation allows to set a bias value between 0 and 1 for relational attributes which modifies the order of attributes to take to partition on. Finally, attributes contain a list attribute called entities, which names all entity types to use for looking for redundant information.
```yaml
attributes:
  id:
    anonymization_type: direct_identifier
  gender:
    type: nominal
    anonymization_type: quasi_identifier
    bias: 0 
  age:
    type: numerical
    anonymization_type: quasi_identifier
    bias: 0
    hierarchy:
      name: '1-100'
      children:
      - name: '1-20'
        children:
          - name: '1-10'
          - name: '11-20'
      - name: '21-40'
        children:
          - name: '21-30'
            children:
              - name: '21-25'
              - name: '26-30'
          - name: '31-40'
            children:
              - name: '31-35'
              - name: '36-40'
      - name: '41-60'
        children:
          - name: '41-50'
          - name: '51-60'
      - name: '61-100'
    entities:
      - DATE
  topic:
    type: nominal
    anonymization_type: quasi_identifier
    entities:
      - TOPIC
  sign:
    type: nominal # type can be nominal, ordinal, numerical, date, or text; default is nominal
    anonymization_type: quasi_identifier
    bias: 0 # bias between 0 and 1, default = 0
    entities:
      - SIGN
  date:
    type: date
    anonymization_type: quasi_identifier
    format: "%d,%B,%Y" # Formats apply to this https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    bias: 0
    entities:
      - DATE
  text:
    type: text
    anonymization_type: text
```

The last section in the configuration file contains all entities which should be considered for anonymization. The section is divided into native (built-in) entities from SpaCy and custom entities which require regular expressions to find sensitive terms.

```yaml
entities:
  native:
    - PERSON
    - NORP
    - FAC
    - ORG
    - GPE
    - LOC
    - PRODUCT
    - EVENT
    - WORK_OF_ART
    - LAW
    - LANGUAGE
    - DATE
    - TIME
    - PERCENT
    - MONEY
    - MAIL
    - URL
    - PHONE
    - POSTCODE
  custom:
    JOB: engineer|scientist|biologist|...
    SIGN: aries|taurus|gemini|cancer|leo|virgo|...
    TOPIC: science|...
```
