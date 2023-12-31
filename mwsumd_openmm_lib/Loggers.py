import csv
import os


class Logger:
    def __init__(self, root):
        self.root = root

    def logData(self, CV, walker, metricName, data, mean_of_data, last_data, scoreMetric):
        os.makedirs(f'{self.root}/reports', exist_ok=True)
        cycle = len(os.listdir(f'{self.root}/trajectories'))
        with open(f'{self.root}/reports/datalogger_METRIC_{CV}.log', 'a') as logFile:
            logFile.write(f'Cycle number: {cycle} '
                          f'Walker: {walker}, Metric: {metricName} '
                          f'All data per frame: {data} '
                          f'Mean of data: {mean_of_data} '
                          f'Last Data: {last_data} '
                          f'Score Metric: {scoreMetric}\n')
            logFile.close()

    #
    # @staticmethod
    # def logSlope(data):
    #     print("Loggers data:")
    #     print(data)
    #     # maxDist = max((dist, value) for dist, value in data.items())
    #     mD = max(dist for dist in data.values())
    #     with open('slope_logs', 'w') as slopeLog:
    #         for frames in range(1, len(data)):
    #             slopeLog.write(str(frames) + "\t" + str(mD))
