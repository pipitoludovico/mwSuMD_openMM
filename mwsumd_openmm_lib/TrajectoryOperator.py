import os
import signal

import MDAnalysis as Mda
from MDAnalysis import transformations

from .Parser import mwInputParser


class TrajectoryOperator(mwInputParser):
    def __init__(self):
        super(mwInputParser, self).__init__()
        self.setting_error = f"File or setting missing. "

    def wrap(self, folder):
        os.chdir(f'{self.folder}/tmp/walker_' + str(folder))
        print('wrapping in ' + os.getcwd())
        ext = ('xtc', 'dcd')
        trajFile = None
        psf = None

        if self.initialParameters['Forcefield'] == 'CHARMM':
            if self.initialParameters['PSF'] is None:
                os.kill(os.getpid(), signal.SIGKILL)
                print(self.setting_error)
                raise FileNotFoundError
            else:
                psf = '../../system/%s' % self.initialParameters['PSF']
        elif self.initialParameters['Forcefield'] == 'AMBER':
            if self.initialParameters['PRMTOP'] is None:
                os.kill(os.getpid(), signal.SIGKILL)
                print(self.setting_error)
                raise FileNotFoundError
            else:
                psf = '../../system/%s' % self.initialParameters['PRMTOP']
        elif self.initialParameters['Forcefield'] == 'GROMOS':
            for new_coords in os.listdir(os.getcwd()):
                if new_coords.startswith(self.initialParameters['Output']) and new_coords.endswith('.tpr'):
                    psf = new_coords
        for trajectory in os.listdir(os.getcwd()):
            if trajectory.startswith(self.initialParameters['Output']) and trajectory.endswith(ext):
                trajFile = trajectory

        u = Mda.Universe(psf, trajFile)
        selection = u.select_atoms(f"{self.initialParameters['Wrap']}")
        ag = u.atoms

        if len(selection.atoms) == 0:
            print("your wrapping selection selected 0 atoms! using protein and name CA instead...")
            selection = u.select_atoms('protein and name CA')

        workflow = (transformations.unwrap(ag), transformations.center_in_box(selection),
                    transformations.wrap(ag, compound='fragments'))
        u.trajectory.add_transformations(*workflow)
        try:
            with Mda.Writer('wrapped.xtc', ag) as w:
                for ts in u.trajectory:
                    w.write(ag)
        except:
            print("\nERROR: Writing wrapped.xtc failed.")
            os.kill(os.getpid(), signal.SIGKILL)

        print("Wrapping " + str(folder) + " successfully completed")
        os.chdir(self.folder)
