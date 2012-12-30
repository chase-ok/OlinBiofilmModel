'''
Created on Dec 30, 2012

@author: chase_000
'''

import numpy as np
import cv2
import pickle

def from_model(model):
    return Result(model.render())

class Result(object):
    
    def __init__(self, cells):
        self.cells = cells
        self.rows, self.columns = cells.shape
        
        self._calculate()
        self._finalize()
        
    def _calculate(self):
        self._calculate_mass()
        self._calculate_contours()
        self._calculate_perimeter()
        self._calculate_heights()
        self._calculate_max_height()
        self._calculate_mean_height()
        self._calculate_row_ffts()
        self._calculate_coverage()
        self._calculate_convexity_defects()
        self._calculate_x_correlations()
        self._calculate_overhang()
    
    def _calculate_mass(self):
        self.mass = np.sum(self.cells)/self.columns

    def _calculate_contours(self):
        self.contours, _ = cv2.findContours(np.copy(self.cells), 
                                            cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)

    def _calculate_perimeter(self):
        self.perimeter = sum(cv2.arcLength(c, True) for c in self.contours)\
                         /float(self.columns)

    def _calculate_heights(self):
        heights = np.zeros(self.columns, dtype=int)
        for row in reversed(range(self.rows)):
            heights[np.logical_and((heights == 0), self.cells[row, :])] = row
        self.heights = heights

    def _calculate_max_height(self, top=0.05):
        heights = np.sort(self.heights)
        self.max_height = np.mean(heights[-np.ceil(top*len(heights)):])

    def _calculate_mean_height(self):
        self.mean_height = np.mean(self.heights)

    def _calculate_row_ffts(self):
        freqs = np.fft.fftfreq(self.columns)
        rows = np.vstack(np.fft.fft(self.cells[row, :])
                         for row in range(self.rows))
        self.row_ffts = freqs, rows

    def _calculate_coverage(self):
        self.coverages = self.cells.sum(axis=1)/float(self.columns)

    def _calculate_convexity_defects(self):
        self.convexity_defects = []
        for contour in self.contours:
            try:
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)
            except:
                continue
            if defects is None: 
                continue

            # defects is an Nx1x4 matrix
            for row in range(defects.shape[0]):
                depth = defects[row, 0, 3]/256.0
                self.convexity_defects.append(depth)

    def _calculate_x_correlations(self):
        distances = range(1, self.columns/2)
        self.x_correlations = []

        for row in range(self.rows):
            if not self.cells[row, :].any(): 
                break

            found = np.zeros(len(distances), dtype=int)
            count = np.zeros_like(found)

            for col in range(self.rows):
                cell = self.cells[row, col]

                for i, distance in enumerate(distances):
                    for direction in [-1, 1]:
                        offset = col + distance*direction
                        if offset < 0 or offset >= self.columns:
                            continue

                        count[i] += 1
                        if self.cells[row, offset] == cell:
                            found[i] += 1

            probabilities = found.astype(float)/count
            self.x_correlations.append((distances, probabilities))
            
    def _calculate_overhang(self):
        self.overhang = np.zeros(self.columns, dtype=int)
        empty_count = np.zeros_like(self.overhang)
        
        for row in range(self.heights.max()):
            alive = self.cells[row] > 0
            self.overhang += empty_count*alive
            empty_count += 1
            empty_count[alive] = 0
    
    def _finalize(self):
        del self.cells

    def dump(self, stream):
        pickle.dump(self, stream, pickle.HIGHEST_PROTOCOL)

    def freq_power_of_x_correlations(self, row):
        _, correl = self.x_correlations[row]
        n = len(correl)
        freqs = np.fft.fftfreq(n)
        power = np.abs(np.fft.fft(correl))
        return freqs[:n/2], power[:n/2]
