# Scenario Based robust control

Run and tested in Python 3.9.12

Edit options.py to choose PRISM max memory usage, and enter PRISM path

use run.py to execute

For full details, see paper.pdf

## Dependencies
- Python packages:
    - numpy 1.23.4
    - tqdm 4.64.1
    - matplotlib 3.5.3
    - scipy 1.9.3
- Java Development Kit
- PRISM (iMDP branch) - https://github.com/davexparker/prism

## To reproduce experiments from the paper:

- UAV Plot: select UAV\_var\_noise with medium noise
- Room heating (main): select steered\_n\_room\_heating with 2 rooms and medium noise
- Room heating (unsteered): select n\_room\_heating with 2 rooms and medium noise
- More general formula: Run test.py 

## To Do:
- Bash file to run all experiments
- Make parameters more user friendly
- Tidy code
- Ensure all options work
