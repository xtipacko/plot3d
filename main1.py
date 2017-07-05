import warnings
warnings.filterwarnings("ignore")

import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plot3d import Plot
from pprint import pprint

def filter_float_tupples(*args):
    try:
        result = []
        for arg in args:
            result.append(float(arg))
        return result
    except:
        return None

datapoints = []
with open('houseprices.csv', 'r') as datafile:
    csv_dict = csv.DictReader(datafile)
    for i,row in enumerate(csv_dict):
        datapoint = filter_float_tupples(row['sqft_living'], 
                                         row['yr_built'], 
                                         row['price'])
        if datapoint and not i % 50:
            datapoints.append(datapoint)

prepared_dps = []
for datapoint in datapoints:
    sqft_living = datapoint[0]
    age         = 2017 - datapoint[1] 
    price       = datapoint[2]
    prepared_dps.append([sqft_living, age, price])


np_datapoints = np.array(prepared_dps, dtype=float)
# pprint(np_datapoints)

array_sqft_living = np_datapoints[:,0]
array_age         = np_datapoints[:,1]
array_price       = np_datapoints[:,2]

mean_sqft_living = np.mean(array_sqft_living)
mean_age         = np.mean(array_age)
mean_price       = np.mean(array_price)


# print(f'Mean sqft_living is {mean_sqft_living}')
# print(f'Mean age is {mean_age}')
# print(f'Mean price is {mean_price}')

#standard deviation
std_sqft_living = np.std(array_sqft_living)
std_age         = np.std(array_age)
std_price       = np.std(array_price)

# def mystd(array,mean):
#     ln = array.size
#     return np.sqrt(np.sum((array-mean)**2) / ln)

# mystd_sqft_living = mystd(array_sqft_living,mean_sqft_living)
# mystd_age         = mystd(array_age,mean_age)
# mystd_price       = mystd(array_price,mean_price)

# print(f'Standard deviation for sqft_living is {std_sqft_living:15}  and my is {mystd_sqft_living:<10}')
# print(f'Standard deviation for age is         {std_age:15} and my is {mystd_age:<10}        ')
# print(f'Standard deviation for price is       {std_price:15}  and my is {mystd_price:<10}      ')

scaled_array_sqft_living = (array_sqft_living - mean_sqft_living) / std_sqft_living
scaled_array_age         = (array_age         - mean_age)         / std_age 
scaled_array_price       = (array_price       - mean_price)       / std_price

bias_vector = np.array([1]*scaled_array_sqft_living.size)

scaled_np_dps  = np.column_stack((bias_vector,
                                  scaled_array_sqft_living, 
                                  scaled_array_age, 
                                  scaled_array_price))

# pprint(scaled_np_dps)

#plotting

plot = Plot(np.array([1,1,1])*5)

plot.ax.scatter(scaled_array_sqft_living,
                scaled_array_age,
                scaled_array_price,
                c='g', marker='.')




#C+Ax1+Bx2 = z


def grad(datapoints, C, A, B):
    m = datapoints.shape[0] # amount of rows in matrix
    parameter_vector = np.array([C,A,B,-1])
    hypotheses_minus_price_vector = datapoints.dot(parameter_vector)
    grad = np.transpose(datapoints[:,0:3]).dot(hypotheses_minus_price_vector) /m
    return grad

def SSE(datapoints, C, A, B):
    m = datapoints.shape[0] # amount of rows in matrix
    parameter_vector = np.array([C,A,B,-1])
    hypotheses_minus_price_vector = datapoints.dot(parameter_vector)
    result = np.sum(hypotheses_minus_price_vector**2) / (2*m)
    return result




C, A, B= 0,0,0
a=0.00001
plt.ion()

xx, yy = np.meshgrid(np.linspace(-5, 5, num=10), np.linspace(-5, 5, num=10))
z = C+A*xx+B*xx
plane = plot.ax.plot_surface(xx, yy, z, color=(0.3,0.7,1,0.5),shade=False) 



def redraw_plane(plane, C, A, B):
    plane.remove()
    xx, yy = np.meshgrid(np.linspace(-5, 5, num=10), np.linspace(-5, 5, num=10))
    z = C+A*xx+B*yy
    plane = plot.ax.plot_surface(xx, yy, z, color=(0.3,0.7,1,0.5),shade=False) 
    return plane
    


for i in range(500000):
    new_grad = grad(scaled_np_dps, C, A, B)
    oldC, oldA, oldB = C, A, B
    C, A, B = np.array([C,A,B]) - a*new_grad  
    if not i % 10000:
        plane = redraw_plane(plane, C,A,B)
        plt.pause(0.001)
        print(f'{f"[{i}]":<6} '
              f'C {C*std_price+mean_price:.6f} '
              f'A {A*std_sqft_living+mean_sqft_living:.6f} '
              f'B {B*std_age+mean_age:.6f} , SSE {SSE(scaled_np_dps, C,A,B):.6f}')

