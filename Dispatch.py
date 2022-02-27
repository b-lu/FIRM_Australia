# Step-by-step analysis to decide the dispatch of flexible energy resources
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability

import datetime as dt
from multiprocessing import Pool, cpu_count

def Flexible(instance):
    """Energy source of high flexibility"""

    year, x = instance
    print('Dispatch works on', year)

    S = Solution(x)

    startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)

    Fcapacity = CPeak.sum() * pow(10, 3) # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)

    for i in range(0, endidx - startidx, timestep):
        flexible[i: i+timestep] = 0
        Deficit, DeficitD = Reliability(S, flexible=flexible, start=startidx, end=endidx) # Sj-EDE(t, j), MW
        if (Deficit + DeficitD).sum() * resolution > 0.1:
            flexible[i: i+timestep] = Fcapacity

    flexible = np.clip(flexible - S.Spillage, 0, None)

    return flexible

def Analysis(x):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Dispatch starts at', starttime)

    # Multiprocessing
    pool = Pool(processes=min(cpu_count(), finalyear - firstyear + 1))
    instances = map(lambda y: [y] + [x], range(firstyear, finalyear + 1))
    Dispresult = pool.map(Flexible, instances)
    pool.terminate()

    Flex = np.concatenate(Dispresult)
    np.savetxt('Results/Dispatch_Flexible{}.csv'.format(scenario), Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    endtime = dt.datetime.now()
    print('Dispatch took', endtime - starttime)

    from Statistics import Information
    Information(x, Flex)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Results/Optimisation_resultx.csv', delimiter=',', skip_header=1)
    Analysis(capacities)