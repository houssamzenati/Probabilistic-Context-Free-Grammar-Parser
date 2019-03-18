# NLP HW2

Code for MVA Probabilistic Context Free Grammar implementation

## Prerequisites.
To run the code, follow those steps:

Install requirements (in the repository):

```
pip3 install -r requirements.txt
```

## Parsing

Perform inference on evaluation set (Warning: will replace current 'evaluation_data.parser_output' and 'evaluation_data.gold' files)

```
sh run.sh --inference
```

Perform evaluation once inference is done:

```
sh run.sh --evaluation
```

Does parsing from .txt file (change sample to any other file)

```
sh run.sh --parse sample.txt
```
