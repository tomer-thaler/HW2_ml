#################################
# Your name: Tomer Thaler
#################################
from random import sample

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
        xs=np.sort(np.random.uniform(0, 1, m))
        ys=[]
        for x in xs:
            if (0<=x<=0.2) or (0.4<=x<=0.6) or (0.8<=x<=1.0):
                p1=0.8
            else:
                p1=0.1
            y=int(np.random.rand()<p1) #y=1 iff a number drawn uni from (0,1) is lower than p1 as needed
            ys.append(y)
        return np.column_stack((xs, ys))

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
        n_steps=max(0, ((m_last - m_first) // step) + 1)
        return_arr = np.zeros((n_steps, 2)) #in each row there will avg empirical error,avg true error for the according row number of samples
        return_arr_idx=0
        for n in range(m_first, m_last+1, step):
            avg_empirical_error=0.0
            avg_true_error=0.0
            for i in range(T):
                sample=self.sample_from_D(n)
                sample_xs,sample_ys = sample[:, 0],sample[:, 1]
                erm_intervals,erm_empirical_error=intervals.find_best_interval(sample_xs, sample_ys,k)
                avg_empirical_error+=(erm_empirical_error/(n*T))
                avg_true_error+=(self.true_error(erm_intervals)/T)
            return_arr[return_arr_idx]=[avg_empirical_error, avg_true_error]
            return_arr_idx+=1

        #at this point computaion are over, now we will show the graphs needed to grasp whats going on
        ms = np.arange(m_first, m_last + 1, step)
        plt.plot(ms, return_arr[:, 0], label='Empirical Error')
        plt.plot(ms, return_arr[:, 1], label='True Error')
        plt.xlabel('Sample size (n)')
        plt.ylabel('Error')
        plt.title(f'ERMs Empirical vs. True error with k={k} (average over {T} runs)')
        plt.legend()
        plt.savefig('ERMs_empirical_vs_true_error_sample_size(n)_dependency.png')
        plt.show()

        return return_arr

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        sample=self.sample_from_D(m)
        sample_xs, sample_ys = sample[:, 0], sample[:, 1]
        num_of_ks=max(0, ((k_last - k_first) // step) + 1)
        record_arr = np.zeros((num_of_ks, 2)) #in each row there will avg empirical error,avg true error for the according row number of k
        record_arr_idx=0
        for curr_k in range(k_first, k_last+1, step):
            empirical_error = 0.0
            true_error = 0.0
            erm_intervals, erm_empirical_error = intervals.find_best_interval(sample_xs, sample_ys, curr_k)
            empirical_error = (erm_empirical_error/m)
            true_error = self.true_error(erm_intervals)
            record_arr[record_arr_idx] = [empirical_error, true_error]
            record_arr_idx += 1
        best_idx = np.argmin(record_arr[:, 0])
        ks = np.arange(k_first, k_last + 1, step)
        best_k = ks[best_idx]
        #at this point record arr is 2d array with num_of_ks rows and 2 colls: (empirical error,true error)
        #now we will print and save a plot of the errors/ks
        plt.plot(ks, record_arr[:, 0], label='Empirical Error')
        plt.plot(ks, record_arr[:, 1], label='True Error')
        plt.xlabel('Number of intervals (k)')
        plt.ylabel('Error')
        plt.title(f'ERMs Empirical vs. True error for increasing k values')
        plt.legend()
        plt.savefig('ERMs_empirical_vs_true_error_num_of_intervals(k)_dependency.png')
        plt.show()
        return best_k




    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        #we start by splitting the data 80/20 randomly
        sample=self.sample_from_D(m)
        validation_error_per_k = np.zeros(10)
        hypotheses = [None] * 10
        indices = np.arange(m)
        np.random.shuffle(indices)
        split = int(0.8 * m)
        train = sample[indices[:split]]
        train = train[train[:, 0].argsort()]  # sorting train by xs
        validation = sample[indices[split:]]
        xs_train, ys_train = train[:, 0], train[:, 1]
        xs_val, ys_val = validation[:, 0], validation[:, 1]
        #finished splitting to train and validation, now we will check different k values

        for k in range(1,11):
            erm_intervals, erm_empirical_error = intervals.find_best_interval(xs_train, ys_train, k)
            hypotheses[k - 1] = erm_intervals
            num_of_validations=len(validation)
            num_of_wrong_labels=0
            for i in range(num_of_validations):
                num_of_wrong_labels+=(self.hipo_guess_for_point(erm_intervals,xs_val[i])!=ys_val[i])
            validation_error_per_k[k-1] = num_of_wrong_labels/num_of_validations
        #finished checking validation errors for different k's now we will locate argmin k
        best_idx = np.argmin(validation_error_per_k[:10])
        best_k = best_idx+1
        best_hypothesis = hypotheses[best_idx]
        # at this point best k holds the k for which the validation error is minimal
        # now we will print and save a plot of the errors/ks
        ks = np.arange(1, 11)
        plt.plot(ks, validation_error_per_k[:10])
        plt.xlabel('Number of intervals (k)')
        plt.ylabel('Validation error')
        plt.title(f'Holdout validation error vs k (m={m})')
        plt.savefig('holdout_validation_error_num_of_intervals(k)_dependency.png')
        plt.show()

        #before returning the best k,
        #below is a printing section that prints the best k and its intervals
        '''
        print(f'Best k: {best_k}')
        clean_intervals = [(float(a), float(b)) for (a, b) in best_hypothesis]
        print(f'Best hypothesis (intervals): {clean_intervals}')
        '''
        return best_k

    #################################
    # Place for additional methods
    def true_error(self,intervals):
        """Finds the true error for hypothesis given as list of intervals.
        Input: intervals - the list of intervals representing h
        Returns: true error of h
        """
        error=0.0
        regions=[(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,1.0)]
        probs=[0.8,0.1,0.8,0.1,0.8] #prob that y=1 given x in according region
        for id,(region_start,region_end) in enumerate(regions):
            prob1=probs[id]
            region_length=region_end-region_start
            region_intersection_intervals=0.0

            for (l,u) in intervals:
                overlap_start=max(region_start,l)
                overlap_end=min(region_end,u)
                if overlap_start<overlap_end:
                    region_intersection_intervals+=overlap_end-overlap_start

            zero_mapped_area_in_region = region_length - region_intersection_intervals #part where h=0
            false_positive_penalty = (1 - prob1) * region_intersection_intervals #h=1 but y=0
            false_negative_penalty = prob1 * zero_mapped_area_in_region #h=0 but y=1
            error += false_positive_penalty + false_negative_penalty
        return error

    def hipo_guess_for_point(self,intervals,point):
        """An indicator function whether point is in union of intervals e.g h(point)
                Input: intervals - the list of intervals representing h, and a point to be considered
                Returns: h(point) 0/1 value
                """
        for region in intervals:
            if region[0]<=point<=region[1]:
                return 1
        return 0

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)



