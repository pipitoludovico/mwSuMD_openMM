import multiprocessing as mp
import time

from mwsumd_openmm_lib.TrajectoryOperator import *
from mwsumd_openmm_lib.Runner_OPENMM import *
from mwsumd_openmm_lib.Utilities import ProcessManager


class Runner(mwInputParser):

    def __init__(self, par):
        self.par = par
        super(mwInputParser, self).__init__()
        self.walk_count = 1


        self.trajCount = len([x for x in os.scandir(f'{str(self.initialParameters["Root"])}/trajectories')])
        self.customProductionFile = None

    def runAndWrap(self):
        walker_snapshot = self.par['Walkers']
        if self.par['Relax'] is True:
            self.par['Walkers'] = 1

        print('mwSuMD is working in ' + os.getcwd())
        print("Trajectory count: " + str(self.trajCount))

        self.runSimulation()

        print("Wrapping results...")
        for i in range(1, self.initialParameters['Walkers'] + 1):
            files = os.listdir(f"./tmp/walker_{i}")
            trajectory = next((file for file in files if file.endswith('.xtc') or file.endswith('dcd')), None)
            if trajectory:
                continue
            else:
                raise Exception("No trajectory found. Check your tmp folder.")

        trajOperator = TrajectoryOperator()
        with mp.Pool() as p:
            p.map(trajOperator.wrap, range(1, self.par['Walkers'] + 1))
        p.close()
        p.join()
        self.par['Walkers'] = walker_snapshot

    def runSimulation(self):
        # let's divide the available GPU in batches by the number of walkers
        manager = ProcessManager()
        GPUs = manager.getGPUids()
        # let's exclude the GPU id if we want to keep a GPU for other jobs
        if self.initialParameters.get("EXCLUDED_GPUS"):
            for excluded in self.initialParameters.get("EXCLUDED_GPUS"):
                GPUs.remove(excluded)
        GPUbatches, idList = manager.createBatches(walkers=self.par['Walkers'], total_gpu_ids=GPUs)
        print('#' * 200)
        if self.initialParameters['Mode'] == 'parallel':
            print("\n\n")
            print('*' * 200)
            print("Running parallel mode")
            runner = RunnerOPENMM()
            manager = mp.Manager()
            q = manager.Queue()
            start_time_parallel = time.perf_counter()
            walk_count = 1
            results = []
            with mp.Pool(processes=len(idList)) as pool:
                for GPUbatch in GPUbatches:
                    for GPU in GPUbatch:
                        results.append(pool.apply_async(runner.runOPENMM, args=(walk_count, GPU)))
                        walk_count += 1
                print(f"Waiting for all processes to finish...")
                for res in results:
                    res.get()
                while not q.empty():
                    q.get()
                print(f"All batches finished.")
            pool.close()
            pool.join()
            self.trajCount += 1
            end_time_parallel = time.perf_counter()
            print(f"Time taken with multiprocessing: {end_time_parallel - start_time_parallel:.2f} seconds")
        else:
            print("Running serial mode")
            start_time_serial = time.perf_counter()
            for GPUbatch in GPUbatches:
                for GPU in GPUbatch:
                    print("we are in " + os.getcwd())
                    runner = RunnerOPENMM()
                    runner.runOPENMM(self.walk_count, GPU)
                    self.walk_count += 1
            end_time_serial = time.perf_counter()
            final_time_serial = end_time_serial - start_time_serial
            print("Serial Final Time:")
            print(final_time_serial)
            self.trajCount += 1

        print("\nMD Runs completed.")
        print('#' * 200)
