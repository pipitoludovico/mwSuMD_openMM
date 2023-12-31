#!/usr/bin/env python3
import os.path

import pandas as pd
from mwsumd_openmm_lib import ArgParser

ArgParser.ArgParser()

if not os.path.exists('./system'):
    print('\nPlease make your ./system folder with the equilibrated system files and outputs')
    exit()

from mwsumd_openmm_lib import Utilities
from mwsumd_openmm_lib import SuMD_openmm

# Get PID:
Utilities.ProcessManager()

print("If you want to use your personal setting for simulating, please, place it the system folder, call it \n"
      "production.inp/namd/mdp (according to your engine) and mwSuMD will use that instead of the default file. \n"
      "If you choose to do so, make sure it points to a folder named 'restart' to look for the restart binaries.\n")


def main():
    sumd = SuMD_openmm.suMD1()
    sumd.run_mwSuMD()
    exit()


if __name__ == '__main__':
    main()
