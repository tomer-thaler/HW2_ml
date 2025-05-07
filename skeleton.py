#################################
# Your name: Tomer Thaler
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO: Implement me
        pass


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods
    def true_error(selfself,intervals):
        error=0.0
        regions=[(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,1.0)]
        probs=[0.8,0.1,0.8,0.1,0.8] #prob that y=1 given x in according region
        for id,(region_start,region_end) in enumerate(regions):
            prob1=probs[id]
            region_length=region_start-region_end
            region_intersection_intervals=0.0
            for (l,u) in intervals:
                overlap_start=max(region_start,l)
                overlap_end=min(region_end,u)
                if overlap_start<overlap_end:
                    region_intersection_intervals+=overlap_end-overlap_start
                zero_mapped_area_in_region=region_length-region_intersection_intervals
                false_positive_penalty=(1-prob1)*region_intersection_intervals
                false_negative_penalty=prob1*zero_mapped_area_in_region
                error+=false_positive_penalty+false_negative_penalty
        return error




    #################################


if __name__ == '__main__':
    print("start\n")
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)


