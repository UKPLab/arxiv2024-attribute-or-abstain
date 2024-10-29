# Attribute or Abstain: Large Language Models as Long Document Assistants

<img src="img/Eyecatcher.drawio.png" width="500">

Jan Buchmann, Xiao Liu, Iryna Gurevych

[Ubiquitous Knowledge Processing Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) (UKP), TU Darmstadt

This repository contains the evaluation and analysis code for LAB, the **L**ong document **A**ttribution **B**enchmark introduced in our paper. 

From the abstract of the paper: 

> LLMs can help humans working with long documents, but are known to hallucinate. Attribution can increase trust in LLM responses: The LLM provides evidence that supports its response, which enhances verifiability. Existing approaches to attribution have only been evaluated in RAG settings, where the initial retrieval confounds LLM performance. This is crucially different from the long document setting, where retrieval is not needed, but could help. Thus, a long document specific evaluation of attribution is missing. To fill this gap, we present LAB, a benchmark of 6 diverse long document tasks with attribution, and experiment with different approaches to attribution on 5 LLMs of different sizes, both prompted and fine-tuned.

Using this repository, the ability of LLMs (in combination with retrievers) to produce attributed responses to long document tasks can be tested. 

To download the pre-formatted datasets, please see [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4276.2).

## Setup

### Data

It is assumed that there is a data folder at `../data` (relative to this repository).
This folder should contain the following:
```
attribution
    datasets <- Datasets to test attributability evaluation models
    results <- Test results for attribution evaluation models
datasets <- Datasets to evaluate attribution capabilities of LLMs 
openai_cost_tracking <- Only needed when using openAI models
results <- All predictions, results and models will be stored here
tensorboard <- Logging
```

You can simply download the data from [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4276.2), unzip it and place it in the same folder as this repository.

### Install requirements
```bash
conda create -n lab_env python=3.11
conda activate lab_env
# There is a requirements conflict because we are using cuda 11.8,
# so  installation is done in this bash script
bash install.sh
```

### Run on your machine

Set correct paths in `config/location/local.yaml` or create new yaml file with your own paths. Use by setting `location=local` when running experiments.

## Quick Start

Running experiments is handled by two scripts that should be used in order: First, [run_evaluation.py](run_evaluation.py) handles LLM training and generation, and optionally pre-generation retrieval, resulting in a list of LLM predictions. Second, these predictions are evaluated by [evaluate_predictions.py](evaluate_predictions.py). This script optionally does post-generation retrieval and finally computes evaluation metrics.

For example, to run a `post-hoc` experiment with ChatGPT on QASPER, we first run [run_evaluation.py](run_evaluation.py):
```bash
# Run inference on QASPER with ChatGPT
python run_evaluation.py description='post_hoc' task=qasper model=gpt-35-turbo-0613 use_dev_as_test_data=False required_aspects=answer_only
```
This will produce a new folder in the results directory, named with a unique hash.
Get the hash and insert it below. This will first do post hoc retrieval and then evaluate with task-specific metrics. 
```bash
python evaluate_predictions.py <hash> --post_hoc_extract
```

Find out more:
- How to evaluate other models is explained [here](#evaluating-more-models)
- How to reproduce the experiments in the paper is explained [here](#reproducing-all-experiments-from-the-paper).
- More details on custom use of the evaluation scripts can be found [here](#how-this-repository-works).

## Reproducing all experiments from the paper

The experiments in the paper can be reproduced by running the scripts [run_evaluation.py](run_evaluation.py), [evaluate_predictions.py](evaluate_predictions.py) and [evaluate_attribution.py](evaluate_attribution.py) with various combinations of parameters.

### Validating Attribution Evaluation Models on annotated data (Table 2)

To validate the attribution evaluation models TRUE, Attrscore and Minicheck on the data annotated during the work on the paper, use the `evaluate_attribution.py` script:

```bash
# Evaluate the TRUE model on QASPER data
python evaluate_attribution.py --description 'true_nli-qasper' --model_name true_nli --task_name qasper --partition dev --is_csv
```

All needed combinations of command line arguments can be found in `experiments/templates/attribution_evaluation_validation.py`. Each line gives the arguments for a single run. Copy a line and paste it to the command line:

````bash
python evaluate_attribution.py <pasted_here>
````

### Main Experiments - Running Attribution Approaches on LAB Benchmark (Table 3)

Running the main experiments involves up to 3 steps:
1. Fine-tuning on the downstream task datasets.
2. Running inference (generating predictions) on the downstream task datasets.
3. Evaluating the predictions.

#### Fine-tuning

To fine-tune a model, use [run_evaluation.py](run_evaluation.py) and pass `do_train=True`:

```bash
# Fine-tune Flan-T5-XL on QASPER with citation
python run_evaluation.py description='finetuning' model=flan-t5-xl task=qasper do_train=true
# Fine-tune Flan-T5-XL on QASPER without citation
python run_evaluation.py description='finetuning' model=flan-t5-xl task=qasper do_train=true required_aspects=answer_only
```

To run all fine-tuning runs from the paper, copy the arguments from  [experiments/templates/flan-t5-xl_fine-tuning.txt](experiments/templates/flan-t5-xl_fine-tuning.txt) and paste them to the command line. Each line defines a single run: 

```bash
python evaluate_attribution.py <paste_here>
```

Note that for fine-tuning on GovReport, we are using a version of GovReport that was auto-annotated with evidence using BM25. From our paper: "To simulate gold evidence, we used BM25 to find the 2 best-matching paragraphs from a document for each sentence in the gold summary".

#### Running Inference (Generating Predictions)

To evaluate models, use [run_evaluation.py](run_evaluation.py) and [run_evaluation.py](run_evaluation.py) (see [Quick Start](#quick-start)). 

First, run the LLMs to generate predictions: To do all runs, get the arguments from the files in [experiments/templates](experiments/templates):
- [post-hoc_runs.txt](experiments/templates/post-hoc_runs.txt)
- [retrieve-then-read_runs.txt](experiments/templates/retrieve-then-read_runs.txt)
- [citation_runs.txt](experiments/templates/citation_runs.txt)
- [reduced-post-hoc_runs.txt](experiments/templates/reduced-post-hoc_runs.txt)
- [reduced-citation_runs.txt](experiments/templates/reduced-citation_runs.txt)

Each line in a file defines a single run. To do evaluation runs with fine-tuned Flan-T5-XL, you will need to replace TODO with the respective hash from the fine-tuning experiment.  

```bash
python run_evaluation.py <paste_here>
```

#### Evaluating Predictions

After having generated predictions, evaluate them. Get the arguments from [experiments/templates/evaluation.txt](experiments/templates/evaluation.txt). There are two lines per dataset: One with post-hoc evidence retrieval (for `post-hoc` and `reduced-post-hoc`, and one without (for `retrieve-then-read`, `citation` and `reduced-citation`). Get the respective hashes from the LLM runs and paste them to replace TODO. Then copy a line from the file and paste it to the command line:

```bash
python evaluate_predictions.py <paste_here>
```

#### Running Experiments in Bulk

This repo provides two bash scripts to conveniently run multiple experiments in bulk.

In [jobs_iter.sh](jobs_iter.sh), you can specifiy an experiment file (e.g. [experiments/templates/post-hoc_runs.txt](experiments/templates/post-hoc_runs.txt)). Each line in this file should contain the arguments for a single run. The script then loops over the lines in the file and runs [run_evaluation.py](run_evaluation.py), [evaluate_predictions.py](evaluate_predictions.py) or [evaluate_attribution.py](evaluate_attribution.py), passing the content from the respective line as parameters. See [jobs_iter.sh](jobs_iter.sh) for usage instructions.

If you have a cluster with SLURM batching, you can use [batch_slurm_job_template.sh](batch_slurm_job_template.sh). In [batch_slurm_job_template.sh](batch_slurm_job_template.sh), specify your experiment file of choice. As above, each line in this file should contain the arguments for a single run. Each array task will then run [run_evaluation.py](run_evaluation.py), [evaluate_predictions.py](evaluate_predictions.py) or [evaluate_attribution.py](evaluate_attribution.py), passing the content from a specific line as parameters. See [batch_slurm_job_template.sh](batch_slurm_job_template.sh) for usage instructions.

## Evaluating your own models

### Evaluating huggingface models

1. Add a new model config for your model. Use the template at `config/model/new_hf_model.yaml`. 
2. Adapt `experiments/templates/your_model_runs.txt` by replacing TODO with the name of your model (the name of the config file without ".yaml")
3. Run inference: See [Running Inference](#running-inference-generating-predictions) and use `your_model_runs.txt` instead of the files for reproducing experiments.
4. Evaluate predictions: See [Evaluating Predictions](#evaluating-predictions)

### Evaluating OpenAI models served via Azure

1. Add a new model config for your model. Use the template at `config/model/new_openai_azure_model.yaml`.
2. Adapt `experiments/templates/your_model_runs.txt` by replacing TODO with the name of your model (the name of the config file without ".yaml")
3. Run inference: See [Running Inference](#running-inference-generating-predictions) and use `your_model_runs.txt` instead of the files for reproducing experiments.
4. Evaluate predictions: See [Evaluating Predictions](#evaluating-predictions)

## Generating plots and tables

The scripts in `analysis/` are used to generate plots and tables. To use them on your setup, follow these steps: 

1. Adjust `analysis/util.infer_base_data_path` to return a path that points to your data directory. 
2. Adjust the `results.csv` in the data directory root: 
   1. If you only ran the models from the paper, adjust the column `hash` to contain the hashes from your runs.
   2. If you added other runs, add new rows to the table as appropriate. The columns with numeric content will be filled automatically. 
   3. If you added new models, add them to the `MODEL_ORDER` and `MODEL_SHORT_NAMES` global variables in `analysis/util`.
3. Run the scripts. The outputs will be put in the `plots` and `tables` directories in your data directory.

## How this repository works

### Important concepts

In our paper, we define 5 approaches to producing attributed responses to long document tasks: 

1. `post-hoc`: Given a long document and a task description, an LLM generates a response. A retriever then retrieves evidence segments for the response from the document.
2. `retrieve-then-read`: Given a task description, a retriever retrieves the most relevant segments from the document. An LLM generates a response based on these segments, and all segments are used as evidence.
3. `citation`: Given a long document and a task description, an LLM generates a response and pointers to evidence segments. 
4. `reduced-post-hoc`: Given a task description, a retriever retrieves relevant segments from the document (more than in `retrieve-then-read`). An LLM generates a response based on these segments. A different retriever then retrieves evidence segments from the document.
5. `reduced-citation`: Given a task description, a retriever retrieves relevant segments from the document (more than in `retrieve-then-read`). An LLM generates a response based on these segments and pointers to evidence segments.

This means there is at least one LLM generation step, and up to two optional retrieval steps. Pre-generation retrieval is done in `retrieve-then-read` and `reduced` approaches, post-generation retrieval is done in `post-hoc` approaches.

Running experiments is handled by two scripts that should be used in order: First, `run_evaluation.py` handles LLM training and generation, and optionally pre-generation retrieval, resulting in a list of LLM predictions. Second, these predictions are further handled by `evaluate_predictions.py`. This script optionally does post-generation retrieval and finally computes evaluation metrics.

### run_evaluation.py

The `run_evaluation.py` script handles LLM training and generation and optionally pre-generation retrieval. The hydra config is used to control all hyperparameters. 

Each unique configuration results in a unique hash that is specific to the run. Running the script results in a new folder in the results directory. After the run, this folder will contain the full config in json format, the predictions made by the model in jsonl format, and a directory called `evaluation`, which contains evaluation-specific files (explained under `evaluate_predictions.py`)

#### The Config

Hyperparameter management is done using the [hydra](https://hydra.cc/) package. All hyperparameters are documented in the corresponding classes in the `config_lib/` directory. These classes document all config parameters and serve as typecheckers for the input values.

The default values of the parameters can be seen in the yaml files in the `config/` directory. The defaults can be overridden by adding `<param_name>=<param_value>` to the command.

##### Important config parameters

All config parameters are documented in the respective module in the `config_lib/` directory. 

- `slurm_job_id`: The id of the slurm job. 
- `description`: A free text description of the experiment (has no effect on the actual run performed)
- `model`: The model to use. See `config/model` for all available models.
- `task`: The task to evaluate on. See `evaluation/tasks` for all available tasks.
- `required_aspects`: str, one of
  - "answer_and_segments" (used when doing `citation`)
  - "answer_only" (used when doing `post-hoc` and `retrieve-then-read`)
- `do_train`: Whether to fine-tune the model on the train set of the task.
- `use_dev_as_test_data`: Whether to evaluate on the dev data.
- `use_first_n_test_instances`: Determine the number of instances to evaluate on. If set to -1, all available test / dev instances are used.

#### Automatic setting of config parameters

Some config parameters need to be set depending on the task and experimental setup. In many cases, this is predictable and does not need to be handled by the user. Instead, this is handled by the `config_lib.base_config.init_config` function. For example, when doing `citation`, we need to add paragraph identifiers ("[1]") to the LLM input, while this is not needed when doing `post-hoc` evidence retrieval. The `is_mode` config parameter is set automatically depending on the value of `required_aspects` to reflect this.

### evaluate_predictions.py

The `evaluate_predictions.py` script handles the evaluation of runs from the `run_evaluation.py` script and optionally does `post-hoc` evidence retrieval. It loads the predictions and config from the given hash and uses task-specific evaluation metrics and models.

Each run of `evaluate_predictions.py` results in a new numbered folder in the `evaluation` directory of the given hash and all outputs are put into this folder, such as:

- `evaluation_config.json`: Contains evaluation parameters
- `results.json`: Contains final evaluation results
- `output_for_annotation.csv`: Contains responses and evidence for attributability annotation
- `predictions.jsonl`: The predictions with responses and evidence. If no re-extraction or `post-hoc` retrieval was done, these are similar to the original predictions from the hash root folder.
- `predictions_<answer_metric>.csv`: Contains the predictions of the model together with task-specific scores for response quality and evidence quality.

The script is controlled through command line arguments: 

```bash
python evaluate_predictions.py <hash> [optional arguments]
```

You can supply one or multiple hashes, but they should be coming from runs on the same task under the same approach. 

#### Important optional command line arguments

- `--post_hoc_extract`: If given, `post-hoc` retrieval is done. The retrieval model and number of retrieved segments are automatically selected depending on the task.
- `--re_extract`: Whether the raw generation from the LLM should be re-parsed (e.g. to extract cited segments). This is useful in cases something went wrong in the original parse. 

## Examples

### Post hoc 
To run post hoc experiments, we first run LLM inference, and then do post hoc evidence retrieval
```bash
# Run inference on QASPER with GPT-4, using the first 100 instances from the test set
python run_evaluation.py description='post_hoc' task=qasper model=gpt-4-turbo-128k use_dev_as_test_data=False required_aspects=answer_only use_first_n_test_instances=100
# This will produce a new folder in the results directory, named with a unique hash
# Get the hash and insert it below. This will first do post hoc retrieval and then evaluate with task-specific metrics. 
python evaluate_predictions.py <hash> --post_hoc_extract
```

### Retrieve-then-read
```bash
# Run inference on QASPER with GPT-4, using the first 100 instances from the test set
python run_evaluation.py description='post_hoc' task=qasper model=gpt-4-turbo-128k use_dev_as_test_data=False required_aspects=answer_only use_first_n_test_instances=100 do_retrieve_then_read=short
# This will produce a new folder in the results directory, named with a unique hash
# Get the hash and insert it below. This will evaluate with task-specific metrics.
python evaluate_predictions.py <hash>
```

### Citation
```bash
# Run inference on QASPER with GPT-4, using the first 100 instances from the test set
python run_evaluation.py description='post_hoc' task=qasper model=gpt-4-turbo-128k use_dev_as_test_data=False required_aspects=answer_and_segments use_first_n_test_instances=100
# This will produce a new folder in the results directory, named with a unique hash
# Get the hash and insert it below. This will evaluate with task-specific metrics.
python evaluate_predictions.py <hash>
```

### Reduced Post Hoc
```bash
# Run inference on QASPER with GPT-4, using the first 100 instances from the test set
python run_evaluation.py description='post_hoc' task=qasper model=gpt-4-turbo-128k use_dev_as_test_data=False required_aspects=answer_and_segments use_first_n_test_instances=100 do_retrieve_then_read=long
# This will produce a new folder in the results directory, named with a unique hash
# Get the hash and evaluate. This will do post hoc retrieval and evaluate with task-specific metrics. 
python evaluate_predictions.py <hash> --post_hoc_extract
```

### Reduced Citation
```bash
# Run inference on QASPER with GPT-4, using the first 100 instances from the test set
python run_evaluation.py description='post_hoc' task=qasper model=gpt-4-turbo-128k use_dev_as_test_data=False required_aspects=answer_and_segments use_first_n_test_instances=100 do_retrieve_then_read=long
# This will produce a new folder in the results directory, named with a unique hash
# Get the hash and insert it below. This will evaluate with task specific metrics.
python evaluate_predictions.py <hash>
```

## Adding new tasks / datasets

- Add new datasets to `../data/datasets`. Create a new directory `<dataset_name>`. In that new directory, create one directory `<dataset_name>_itg` for the dataset in itg format and optionally another directory for the raw dataset.
- Add a new `<task-name>.py` file to `evaluation/tasks`. Implement the required classes (`...Instance`, `...Prediction`, `...Result`, `...Task`) in that file, inheriting from the `Base...` classes in `evaluation/common.py`. You can use the other tasks for reference
- If your dataset exists as individual json files, you might want to use `evaluation/common/SingleFileDataset` to implement data loading
- Add your new classes to `run_evaluation.py`
- Add new yaml file to `config/tasks` that specifies hyperparameters for the task. If new hyperparameters are necessary, add them to `config_lib/base_task/BaseTaskConfig`
- Add new `task_explanation` and `example` to `config/prompts.py`

## Logging

The experiments are automatically logged using a tensorboard logger. This includes the train loss and the metrics from intermediary evaluations on the development set. 

To view the logs in your browser, follow these steps: 

1. Install tensorboard on your local machine `pip install tensorboard`
2. Set logging directory in `config/location/local`
3. Start tensorboard `tensorboard --logdir /path/to/logging/dir`

## Structure of this repository
```
analysis/ <- contains scripts to make analyses, plots and tables

attribution_eval/
    attribution_dataset.py <- Contains code to construct claims from responses for attributability evaluation
    attribution_model.py <- Contains code for the attributability evaluation models
    util.py <- utility functions 

config/ <- contains yaml files with default values for parameters
    location/ <- Default paths for models, result files, predictions, datasets, ...
    model/ <- defaults for currently implemented models
    task/ <- defaults for currently implemented tasks
    config.yaml <- top-level config file with general parameters
    prompts.yaml <- Hard coded prompt elements to choose from
    
config_lib/ <- contains config classes for documentation and type checking

dataset_creation/ 
    make_attributed_dataset.py <- Contains code to automatically annotate evidence via post hoc retrieval

evaluation/
    tasks/ <- Contains task classes that implement instance loading and evaluation
    callbacks.py <- Contains callbacks that are called during training and evaluation, important for storing predictions and logging
    common.py <- Base classes for tasks, instances, predictions. New tasks should inherit from these base classes.
    metrics.py <- Metrics for evaluation (e.g. Answer F1)
    run.py <- Contains run() which is called in run_evaluation.py
    validator.py <- Basic implementation of evaluation loop.
    util.py <- Utility functions for evaluation
    
models/
    api_lm.py <- General implementation for API-served models (e.g. ChatGPT).
    base_model.py <- Base class for models
    causal_lm.py <- General implementation of decoder-only models that can be accessed through Huggingface's AutoModelForCausalLM class.
    oracle.py <- A simple oracle model that returns the ground truth as predictions. Can be useful for debugging.
    retrieve.py <- Retrievers for reducing, retrieve then read and post hoc
    seq2seq_lm.py <- General implementation of encoder-decoder LMs.
    
    
notebooks/ <- Ipython notebooks for experimentation.

scripts/
    nq_filter.py <- Script to filter the natural questions dataset

structformer/
    collation_utils.py <- Helper functions for data collation
    example_formatting.py <- Helper functions for formatting of in-context examples.
    format_enforcement.py <- Creates functions for constrained generation
    input_preparation.py <- Functions for data processing
    input_sequence.py <- Functions to produce a single string from an Intertext Document
    sequence_alignment.py <- Functions to map between tokenized ITGs and ITGs.
    
run_evaluation.py <- Main script for experiments
evaluate_predictions.py <- Main script to evaluate runs, e.g. for attributability. Post hoc retrieval is also done here
evaluate_attribution.py <- Main script to evaluate attributability evaluation models

single_slurm_job.sh <- Job script for slurm jobs
```

## Citation

If you use our data or code in your research, please cite

@misc{buchmann2024attributeabstainlargelanguage,
      title={Attribute or Abstain: Large Language Models as Long Document Assistants}, 
      author={Jan Buchmann and Xiao Liu and Iryna Gurevych},
      year={2024},
      eprint={2407.07799},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.07799}, 
}

## Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.