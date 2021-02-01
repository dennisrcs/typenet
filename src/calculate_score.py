import numpy as np
import torch
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
import csv

class CalculateScore:
    """
    Calculates test score
    """
    def __init__(self):
        self.print_scores_to_file = False

    def calculate(self, output, enrollment_size, gallery_size, epoch, part):
        genuine_gallery_test = output[0:enrollment_size,:]
        genuine_gallery = genuine_gallery_test[0:gallery_size,:]
        genuine_test = genuine_gallery_test[gallery_size:,:]

        impostor_test = output[enrollment_size:,:]

        genuine_scores = []
        impostor_scores = []
        for i in range(0, len(genuine_test)):
            score = self.calculateScore(genuine_test[i], genuine_gallery, gallery_size)
            genuine_scores.append(score)

        for i in range(0, len(impostor_test)):
            score = self.calculateScore(impostor_test[i], genuine_gallery, gallery_size)
            impostor_scores.append(score)

        if self.print_scores_to_file:
            with open('genuine_' + epoch + '_' + str(part) + '.csv', 'w') as filehandle:
                for listitem in genuine_scores:
                    filehandle.write('%.3f\n' % listitem)

            with open('impostor_' + epoch + '_' + str(part) + '.csv', 'w') as filehandle:
                for listitem in impostor_scores:
                    filehandle.write('%.3f\n' % listitem)

        stats = get_eer_stats(genuine_scores, impostor_scores)

        return stats.eer
    
    def calculateScore(self, sample, gallery, gallery_size):
        repeated_sample = sample.repeat(gallery_size, 1)
        scores = self.edistance(repeated_sample, gallery)

        return (scores.sum() / gallery_size)
            
    def edistance(self, a, b):
        a = a.to(torch.device('cpu'))
        b = b.to(torch.device('cpu'))

        return np.linalg.norm(a-b, axis=1)
