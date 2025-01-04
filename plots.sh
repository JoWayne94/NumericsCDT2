#!/bin/sh
# This script is used to plot all the figures included in the report.

python plot.py
cd test_cases/Helmholtz2d/soton_fire
python Helmholtz2d.py
cd ../..
cd ADR2d/soton_fire
python steadyADR2d.py
python ADR2d.py
python ADR2d_2.py
