import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    """Reads draglift data"""

    with open(filename, 'r') as f:
        lines = f.readlines()
    items = [[float(item) for item in line.strip().split()]
             for line in lines[5:]]
    # print(items)
    return np.array(items)


if __name__ == '__main__':
    data = read_data('DragLift_case_6.fce')
    print('Data shape is', data.shape)
    steady_index = np.argmin(np.abs(data[:, 0]-43))
    data = data[steady_index:, :]
    for i in range(data.shape[0]):
        if data[i, 6] > data[-1, 6]:
            full_index = i
            break
    data = data[full_index:, :]

    plt.plot(data[:, 0], data[:, 6])
    plt.grid()
    plt.xlabel('$l/u_{inf}$')
    plt.show()


    print(data[0, 0])

    maxes = []
    mins = []
    for i in range(1, data.shape[0]-1):
        if data[i, 6] > data[i-1, 6] and data[i, 6] > data[i+1, 6]:
            maxes.append(data[i, 6])
            print(f'Max: {data[i,6]} at t={data[i,0]}')
        if data[i, 6] < data[i-1, 6] and data[i, 6] < data[i+1, 6]:
            mins.append(data[i, 6])
            print(f'Min: {data[i,6]} at t={data[i,0]}')
    print(maxes)
    print(mins)
    print(
        f'Starting time is {data[0,0]} at index {int(data[0,0]/0.005)}')
    starting_index = int(data[0,0]/0.005)


    data = read_data('DragLift_case_6.fce')
    steady_points = 10000-starting_index
    periods = 4
    for i in range(1, 5):
        end_ind = starting_index + int(i/periods*steady_points)
        plt.plot(data[starting_index:end_ind, 0], data[starting_index:end_ind, 6])
        plt.grid()
        print(end_ind-(starting_index + int((i-1)/periods*steady_points)))
        plt.show()
