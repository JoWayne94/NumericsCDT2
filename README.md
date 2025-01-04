This is a repository for the MFC CDT Numerical Analysis assignment 2. It contains a linear finite element method written in Python.

### Setup

Python 3.7 or higher is required. The libraries required to be installed are also listed in `requirements.txt` with their tested version numbers provided.

### Generating figures 

The scripts generating plots and figures are located in the `test_cases` directory and `plot.py`. In order to obtain them, run the following shell script:

```sh
$ chmod +x plots.sh
$ ./plots.sh
```

The figures will pop up on your screen, but not saved to your local machine; you will have to close them manually to move forward the runs, or save them if needed. If you close them relatively quickly, the total time for all runs should take around 5 minutes.

### Others

If you want to run individual simulation, the `main.py` and `setup.py` scripts are a good starting point and they act as templates for you to customise your test case using different setup specifications. Otherwise, you can also navigate to the `test_cases_` directories to look at how various test cases are set up, and if you want to try them, just run the script that is not `setup.py`. For example:  

```sh
$ cd test_cases/ADR1d/case_1/

$ python ADR1d.py
```

The configuration of the simulation can be altered in the `setup.py` scripts that are located in the same directory as the respective python test scripts.

### Contact details
Please contact Jo Wayne Tan (jwt617@ic.ac.uk) for any questions or issues.
