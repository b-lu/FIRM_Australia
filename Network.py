# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Transmission(solution, output=False):
    """TDC = Network.Transmission(S)"""

    Nodel, PVl, Windl = (solution.Nodel, solution.PVl, solution.Windl)
    intervals, nodes = (solution.intervals, solution.nodes)

    MPV, MWind = map(np.zeros, [(nodes, intervals)] * 2)
    for i, j in enumerate(Nodel):
        MPV[i, :] = solution.GPV[:, np.where(PVl==j)[0]].sum(axis=1)
        MWind[i, :] = solution.GWind[:, np.where(Windl==j)[0]].sum(axis=1)
    MPV, MWind = (MPV.transpose(), MWind.transpose()) # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MBaseload = solution.GBaseload # MW
    CPeak = solution.CPeak # GW
    pkfactor = np.tile(CPeak, (intervals, 1)) / CPeak.sum()
    MPeak = np.tile(solution.flexible, (nodes, 1)).transpose() * pkfactor # MW

    MLoad = solution.MLoad # EOLoad(t, j), MW

    defactor = MLoad / MLoad.sum(axis=1)[:, None]
    MDeficit = np.tile(solution.Deficit, (nodes, 1)).transpose() * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MWind
    spfactor = np.divide(MPW, MPW.sum(axis=1)[:, None], where=MPW.sum(axis=1)[:, None]!=0)
    MSpillage = np.tile(solution.Spillage, (nodes, 1)).transpose() * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    pcfactor = np.tile(CPHP, (intervals, 1)) / sum(CPHP) if sum(CPHP) != 0 else 0
    MDischarge = np.tile(solution.Discharge, (nodes, 1)).transpose() * pcfactor # MDischarge: DPH(j, t)
    MCharge = np.tile(solution.Charge, (nodes, 1)).transpose() * pcfactor # MCharge: CHPH(j, t)

    MP2V = np.tile(solution.P2V, (nodes, 1)).transpose() * pcfactor # MP2V: DP2V(j, t)

    CDP = solution.CDP
    pcfactorD = np.tile(CDP, (intervals, 1)) / sum(CDP) if sum(CDP) != 0 else 0
    MChargeD = np.tile(solution.ChargeD, (nodes, 1)).transpose() * pcfactorD # MChargeD: CHD(j, t)

    MImport = MLoad + MCharge + MChargeD + MSpillage \
              - MPV - MWind - MBaseload - MPeak - MDischarge + MP2V - MDeficit # EIM(t, j), MW

    FQ = -1 * MImport[:, np.where(Nodel=='FNQ')[0][0]] if 'FNQ' in Nodel else np.zeros(intervals)
    AS = -1 * MImport[:, np.where(Nodel=='NT')[0][0]] if 'NT' in Nodel else np.zeros(intervals)
    SW = MImport[:, np.where(Nodel=='WA')[0][0]] if 'WA' in Nodel else np.zeros(intervals)
    TV = -1 * MImport[:, np.where(Nodel=='TAS')[0][0]]

    NQ = MImport[:, np.where(Nodel=='QLD')[0][0]] - FQ
    NV = MImport[:, np.where(Nodel=='VIC')[0][0]] - TV

    NS = -1 * MImport[:, np.where(Nodel=='NSW')[0][0]] - NQ - NV
    NS1 = MImport[:, np.where(Nodel=='SA')[0][0]] - AS + SW
    assert abs(NS - NS1).max()<=0.1, print(abs(NS - NS1).max())

    TDC = np.array([FQ, NQ, NS, NV, AS, SW, TV]).transpose() # TDC(t, k), MW

    if output:
        MStorage = np.tile(solution.Storage, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
        MDischargeD = np.tile(solution.DischargeD, (nodes, 1)).transpose() * pcfactorD  # MDischargeD: DD(j, t)
        MStorageD = np.tile(solution.StorageD, (nodes, 1)).transpose() * pcfactorD  # SD(t, j), MWh
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind,MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage, solution.MP2V = (MDischarge, MCharge, MStorage, MP2V)
        solution.MDischargeD, solution.MChargeD, solution.MStorageD = (MDischargeD, MChargeD, MStorageD)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC