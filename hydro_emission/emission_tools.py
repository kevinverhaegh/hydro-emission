# Function: create_interpolations
import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
from mat73 import loadmat
from copy import deepcopy
import scipy.io as sciio
import math


def get_default_file_dir():
    # Placeholder implementation for the demo
    return 'hydro_emission/helper_data'

def amjuel_tables(type_amj, table, ne, te,
                  fpath=get_default_file_dir() + '/PEC_data/get_amjuel.mat',
                  clip=True):
    def h4_rate(h4_table, ne_v, Te_v, amj):
        try:
            amj["AMJ"][h4_table]["table"]
        except KeyError:
            rate = math.nan
            return rate

        ne_v = ne_v / 1e6
        temp = 0
        for i in range(0, 9):
            for j in range(0, 9):
                temp = temp + amj["AMJ"][h4_table]["table"][i][j] * (np.log(Te_v) ** i) * (np.log(ne_v / 1e8) ** j)
        rate = 1e-6 * np.exp(temp)
        return rate

    def h2_rate(h2_table, Te, amj):
        try:
            amj["AMJ"][h2_table]["table"]
        except KeyError:
            rate = math.nan
            return rate

        temp = 0
        for i in range(0, 9):
            temp = temp + amj["AMJ"][h2_table]["table"][i] * (np.log(Te) ** i)
        rate = 1e-6 * (np.exp(temp))
        return rate

    def h11_rate(h11_table, Te, amj):
        try:
            amj["AMJ"][h11_table]["table"]
        except KeyError:
            ratio = math.nan
            return ratio

        total = 0
        for i in range(0, 9):
            total = total + (amj["AMJ"][h11_table]["table"][i] * (np.log(Te) ** i))
        ratio = np.exp(total)
        return ratio

    def h12_rate(h12_table, ne_v, Te, amj):
        try:
            amj["AMJ"][h12_table]['table']
        except KeyError:
            print('key not found')
            ratio = math.nan
            return ratio

        ne_v = ne_v / 1e6
        total = np.zeros(np.shape(ne_v))
        for i in range(0, 9):
            for j in range(0, 9):
                total = total + amj["AMJ"][h12_table]["table"][i][j] * (np.log(Te) ** i) * (np.log(ne_v / 1e8) ** j)
        ratio = np.exp(total)
        return ratio

    mat = loadmat(fpath)

    if clip:
        # limit grid for YACORA interpolation
        te = np.clip(te, 0.1, 20)
        ne = np.clip(ne, 5e18, 1.5e20)

    if type_amj == "h4":
        sigmav = h4_rate(table, ne, te, mat)
        return sigmav
    elif type_amj == "h2":
        sigmav = h2_rate(table, te, mat)
        return sigmav
    elif type_amj == "h11":
        sigmav = h11_rate(table, te, mat)
        return sigmav
    elif type_amj == "h12":
        sigmav = h12_rate(table, ne, te, mat)
        return sigmav

def create_interpolations(D_ADAS=None, low_Te_EIR=True,
                          ADAS_lowTe=loadmat(get_default_file_dir() + '/PEC_data/ADAS_EIR_lowTe.mat')):
    if D_ADAS is None:
        D_ADAS = sciio.loadmat(get_default_file_dir() + '/PEC_data/DeuteriumADAS.mat')
    interpolations = {}
    count = 1
    tem, nem = np.meshgrid(D_ADAS['AdasTe'], D_ADAS['AdasNe'])

    for _ in D_ADAS['BalmerExcitation']:
        interpolations['n' + str(count) + 'exc'] = (LinearNDInterpolator(
            (np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
            np.ravel(-6 + np.log10(np.transpose((np.squeeze(D_ADAS['BalmerExcitation'][count - 1, :, :])))))))
        if not low_Te_EIR:
            interpolations['n' + str(count) + 'rec'] = (LinearNDInterpolator(
                (np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
                np.ravel(-6 + np.log10(np.transpose((np.squeeze(D_ADAS['BalmerRecombination'][count - 1, :, :])))))))
        count += 1

    if not low_Te_EIR:
        interpolations['ACD'] = (LinearNDInterpolator(
            (np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
            np.ravel(-6 + np.log10((np.squeeze(D_ADAS['ACDHydrogen']))))))

    interpolations['SCD'] = (
        LinearNDInterpolator((np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
                             np.ravel(-6 + np.log10((np.squeeze(D_ADAS['SCDHydrogen']))))))

    interpolations['CCD'] = (
        LinearNDInterpolator((np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
                             np.ravel(-6 + np.log10((np.squeeze(D_ADAS['CCDHydrogen']))))))

    interpolations['PRB'] = (
        LinearNDInterpolator((np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
                             np.ravel(-6 + np.log10((np.squeeze(D_ADAS['PRBHydrogen']))))))

    interpolations['PLT'] = (
        LinearNDInterpolator((np.transpose([np.ravel(np.log10(np.squeeze(nem))), np.ravel(np.log10(np.squeeze(tem)))])),
                             np.ravel(-6 + np.log10((np.squeeze(D_ADAS['PLTHydrogen']))))))

    if low_Te_EIR:
        PECrec = ADAS_lowTe['PECrec']
        ACD = ADAS_lowTe['ACD']
        wl = []
        wl_keys = []
        for key in PECrec.keys():
            if 'ADAS_' in key:
                wl_keys.append(key)
                wl.append(PECrec[key]['RC']['lambda'])
        indx = np.argsort(wl)[::-1]

        count = 1
        for i in range(0, len(indx)):
            interpolations['n' + str(count) + 'rec'] = (LinearNDInterpolator(
                (np.transpose([np.ravel(np.log10(np.squeeze(PECrec[wl_keys[indx[i]]]['RC']['ne']))),
                               np.ravel(np.log10(np.squeeze(PECrec[wl_keys[indx[i]]]['RC']['te'])))])),
                np.ravel(-6 + np.log10((np.squeeze(PECrec[wl_keys[indx[i]]]['RC']['PEC']))))))
            count += 1
        interpolations['ACD'] = (LinearNDInterpolator(
            (np.transpose([np.ravel(np.log10(np.squeeze(ACD['ne']))), np.ravel(np.log10(np.squeeze(ACD['te'])))])),
            np.ravel(-6 + np.log10((np.squeeze(ACD['ACD'][:,:,1]))))))

    return interpolations


# Function: TECPEC_Yacora
def TECPEC_Yacora(type_yac, N, Ne, Tev, TiHmP, TiHpP):
    data = sciio.loadmat(get_default_file_dir() + '/PEC_data/Yacora_data_' + type_yac + '.mat')

    if len(np.shape(Tev)) > 1:
        MD = True
        shape = np.shape(Tev)
        Ne = np.ravel(Ne)
        Tev = np.ravel(Tev)
        TiHmP = np.ravel(TiHmP)
        TiHpP = np.ravel(TiHpP)
    else:
        MD = False

    Ne = np.clip(Ne, np.min(data['nel']), np.max(data['nel']))
    Tev = np.clip(Tev, np.min(data['Te']), np.max(data['Te']))
    if not isinstance(N, list):
        N = [N]

    N = [x - 1 for x in N]

    PECv = np.zeros([len(N), len(Ne)]) + np.nan
    Isel = Ne * Tev
    if np.shape(TiHmP) == np.shape(Ne):
        Isel = Isel * TiHmP
    if np.shape(TiHpP) == np.shape(Ne):
        Isel = Isel * TiHpP

    Isel = np.logical_not(np.isnan(Isel))

    if type_yac != 'HmHp' and type_yac != 'HmH2p':
        for i in range(0, len(N)):
            coords = []
            for j in range(0, len(data['nel'])):
                for k in range(0, len(data['Te'])):
                    coords.append((np.log10(data['nel'][j][0]), np.log10(data['Te'][k][0])))
            PECv[i, Isel] = griddata(coords, np.ravel(data['PEC'][:, :, N[i]]), (np.log10(Ne[Isel]), np.log10(Tev[Isel])), method='linear')
    elif type_yac == 'HmHp':
        TiHmP = [element * 11600 for element in TiHmP]
        TiHpP = [element * 11600 for element in TiHpP]
        TiHmP = np.clip(TiHmP, np.min(data['TiHmP'][0]), np.max(data['TiHmP'][0]))
        TiHpP = np.clip(TiHpP, np.min(data['TiHpP'][0]), np.max(data['TiHpP'][0]))

        [TiHmPp, TiHpPp] = np.meshgrid(data['TiHmP'], data['TiHpP'])
        p = griddata((np.ndarray.flatten(TiHmPp), np.ndarray.flatten(TiHpPp)), np.ndarray.flatten(data['AdjF']),
                     (TiHmP[Isel], TiHpP[Isel]), method='linear')
        PECv = np.zeros((len(N), len(Ne)))
        PECv[PECv == 0] = np.nan
        for i in range(0, len(N)):
            coords = []
            for j in range(0, len(data['Te'])):
                for k in range(0, len(data['nel'])):
                    coords.append((np.log10(data['nel'][k][0]), np.log10(data['Te'][j][0])))
            PECv[i, Isel] = p * griddata(coords, np.ravel(data['PEC'][:, :, N[i]]), (np.log10(Ne[Isel]), np.log10(Tev[Isel])),
                                         method='linear')
    elif type_yac == 'HmH2p':
        TiHmP = [element * 11600 for element in TiHmP]
        TiHpP = [element * 11600 for element in TiHpP]
        TiHmP = np.clip(TiHmP, np.min(data['TiHm']), np.max(data['TiHm']))
        TiHpP = np.clip(TiHpP, np.min(data['TiH2p']), np.max(data['TiH2p']))

        PECv = np.zeros((len(N), len(Ne)))
        PECv[PECv == 0] = np.nan

        points = (data['nel'], data['TiHm'], data['TiH2p'], data['Te'])
        pnts1 = np.log10([val for sublist in points[0] for val in sublist])
        pnts2 = np.log10([val for sublist in points[1] for val in sublist])
        pnts3 = np.log10([val for sublist in points[2] for val in sublist])
        pnts4 = np.log10([val for sublist in points[3] for val in sublist])
        for i in range(0, len(N)):
            coords = []
            for j in range(0, len(pnts1)):
                for k in range(0, len(pnts2)):
                    for l in range(0, len(pnts3)):
                        for m in range(0, len(pnts4)):
                            coords.append((pnts1[j], pnts2[k], pnts3[l], pnts4[m]))
            PECv[i, Isel] = griddata(coords, np.ravel(data['PEC'][:, :, :, :, N[i]]),
                                     (np.log10(Ne[Isel]), np.log10(TiHmP[Isel]), np.log10(TiHpP[Isel]), np.log10(Tev[Isel])), method='linear')

    if MD:
        PECv_c = deepcopy(PECv)
        PECv = np.zeros(np.insert(shape, 0, len(N)))
        for j in range(0, len(N)):
            PECv[j, :] = np.reshape(PECv_c[j, :], shape)

    return PECv


# Function: TECPEC_Yacora_eff
def TECPEC_Yacora_eff(N, Ne, Tev, TiHmP=None, TiHpP=None, TiH2pP=None, fH2p_mod=0.95, fHm_mod=0.7):
    PEC_H2 = TECPEC_Yacora('H2', N, Ne, Tev, [], [])
    PEC_H2p = TECPEC_Yacora('H2p', N, Ne, Tev, [], [])

    if not TiHmP:
        TiHmP = 2.2 * np.ones(np.shape(Tev))  # Franck Condon energy
    if not TiHpP:
        TiHpP = Tev  # assume Te = Ti
    if not TiH2pP:
        TiH2pP = 2.2 * np.ones(np.shape(Tev))

    PEC_HmHp = TECPEC_Yacora('HmHp', N, Ne, Tev, TiHmP, TiHpP)
    #PEC_HmH2p = TECPEC_Yacora('HmH2p', N, Ne, Tev, TiHmP, TiH2pP)
    PEC_HmH2p=0

    fH2p = fH2p_mod * amjuel_tables('h12', 'H12_2_0c', Ne, Tev)
    fHm = fHm_mod * amjuel_tables('h11', 'H11_7_0a', Ne, Tev)

    PECv = PEC_H2 + PEC_H2p * fH2p + PEC_HmHp * fHm + PEC_HmH2p * fH2p * fHm

    return PECv

def interp(V, A, B, extrap=False, Alim=None, Blim=None):
    # helper function for 2D interpolations which chucks out NaNs for faster computation
    if Blim is None:
        Blim = []
    if Alim is None:
        Alim = []
    import copy
    AA = copy.deepcopy(A)
    BB = copy.deepcopy(B)
    if extrap:
        AA = np.clip(A, np.nanmin(Alim), np.nanmax(Alim))
        BB = np.clip(B, np.nanmin(Blim), np.nanmax(Blim))
    I = np.logical_not(np.isnan(np.ravel(np.log10(AA) * np.log10(BB))))
    out = np.zeros(np.shape(np.ravel(AA))) + np.nan
    out[I] = 10 ** V(np.log10(np.ravel(AA)[I]), np.log10(np.ravel(BB)[I]))

    out = np.reshape(out, np.shape(AA))
    return out

def Balmer_line_emissivity(N,ne,nD,nD2,nDp,Te):
    #calculates hydrogen emissivity (photons/m^3) based on electron density, neutral atom density, neutral molecular density, ion density and electron temperature
    adas = create_interpolations()
    PEC_rec = interp(adas['n' + str(N - 2) + 'rec'], ne,Te)
    PEC_exc = interp(adas['n' + str(N-2) + 'exc'], ne, Te)
    PEC_mol = TECPEC_Yacora_eff(N-1,ne,Te)
    emissivity = ne*nDp*PEC_rec + ne*nD*PEC_exc + nD2*ne*PEC_mol
    return emissivity