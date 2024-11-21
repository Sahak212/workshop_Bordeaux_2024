# ===============================================================
# MODEL WITH THERMODYNAMIC CONSTRAINS AND DYNAMIC GRWOTH YIELD
# Model is designed for modeling of data from Maria about thermophilic and syntrophic propionate and acetate oxidizers, doi: https://doi.org/10.1038/s41396-023-01504-y
# model is designed for batch assay
# in comparison to other model, here Ks values are fixed. This was done because 1. to improve parameter optimizaiton process there is a need to reduce the parameters
# 2. Ks values obtained after optimization first time were very close to values of those parameters in ADM1 model
# ===============================================================
import numpy
import plotly.graph_objs as go
import os
import sys
import io
from scipy.integrate import odeint
from scipy.optimize import *
from sklearn.metrics import mean_squared_error
import pandas as pd
from plotly.subplots import make_subplots
from numpy import log10
from plotly.subplots import make_subplots
import math

# initial parameters: pressure in bar, temperature in K, ΔG in kj/mol, concentration in mol/l, volume in liters, universal gas constant in L * bar / (K * mol), time in hours
initial_parameter = {'Temperature_ex': 25 + 273, 'Temperature_0': 298.15,
                     'R': 0.08314, 'head_space': 0.5,
                     'reaction_media_volume': 0.5, 'duration_of_experiment': 630, 'time_step': 0.05,
                     'Propionate_M': 0.0363, 'Acetate_M':0.0000001, 'H2_partial_pressure': 0.000002, 'CO2_patial_pressure': 0.344, 'N2_partial_pressure': 1.376,
                     'CH4_partial_pressure': 0.000001, 'maximal_pressure': 2, 'inititla_total_pressure': 1.3,
                     'CO_partial_pressure': 1e-6,
                     'qs_max_Prop': 0.01518, 'Ks_Prop': 0.00179, 'Inhib_Prop': 2.19E-07,
                     'qs_max_H2': 1.123e-01, 'Ks_H2': 3.586e-05,
                     'HCO3_in_M': 1e-6, 'Growth_yield_H2': 0.55, 'Growth_yield_Prop': 1.02}


# create function that calculates stoechimetric coefficients of propionate oxidizers from growth yield.
def stoech_Prop(Yxs):
    if Yxs == 0:
        return dict(H20=0, AC=0, HCO3=0, NH4=0, CO2=0, H2=0, Biomass=0)
    else:
        lambdaC = 24.6 / Yxs - 0.3
        PropC = lambdaC + 0.3
        H2O = (-2 * lambdaC + 0.2) / PropC
        AC = lambdaC / PropC
        HCO3 = 0.1 / PropC
        NH4 = -0.2 / PropC
        CO2 = (lambdaC - 0.2) / PropC
        H2 = 3 * lambdaC / PropC
        Bi = 24.6 / PropC
        return dict(H20=H2O, AC=AC, HCO3=HCO3, NH4=NH4, CO2=CO2, H2=H2, Biomass=Bi)


# function that calculates stoechimetric coefficients of hydrogenotrophic archeae
def stoech_H2(Yxs):
    if Yxs == 0:
        return dict(H20=0, HCO3=0, NH4=0, CO2=0, CH4=0, Biomass=0)
    else:
        lambdaC = 24.6 / Yxs - 2.1
        H2_C = lambdaC + 2.1
        H2O = (0.5 * lambdaC + 1.7) / H2_C

        HCO3 = -0.2 / H2_C
        NH4 = -0.2 / H2_C
        CO2 = (-0.25 * lambdaC - 0.8) / H2_C
        CH4 = 0.25 * lambdaC / H2_C
        Biomass = 24.6 / H2_C
        return dict(H20=H2O, HCO3=HCO3, NH4=NH4, CO2=CO2, CH4=CH4, Biomass=Biomass)


# this function filters out the the time points from integrated data, which are present in test data
# things become complicated as time points in integration are calculated by mathematical operation on float numbers, and because of that they did not match data from test data
# example time point equal to 10 in integration is not 10 but rather 9.99999998.
def filter(modeled_data, experiment_data):
    output = []
    for time_point in experiment_data['Time'] * 24:
        index = int(time_point / initial_parameter['time_step'])
        output.append(modeled_data[index])
    return output

State_index = ['C_H2_M', 'X_H2_gX', 'X_Prop_gX', 'C_CO2_M', 'C_propionate_M', 'C_CH4_M', 'C_H2_gas_M', 'C_CH4_gas_M', 'C_CO2_gas_M', 'Dead_Biomass', 'C_acetate_M']
Id = {s: ID_s for ID_s, s in enumerate(State_index)}

# ================================================
# Main function to realize the model
# ================================================

def integral(param):
    # optimizable parameters for reaction kinetics
    qs_max_H2 = param[2]  # maximum substrate uptake rate of hydrogenotrophs
    Ks_H2 = param[3] # "affinity" of the hydrogenotrophs
    k_L_a = 8.33333  # units in h^-1, rate of liquid-gas transfer
    qs_max_Prop = param[4]  # maximum substrate uptake rate of the acetate oxidizers
    Ks_Prop = param[5]  # "affinity" of the acetate oxidizers

    # Gibbs dissipation energy startion values (ΔG0). Those values are calculated for standart conditions with temperature correction with Van 't Hoff equation
    # the calcualtion was done according to stoechimetric coefficients from file Formulas_acetate.pdf with pyCHNOSZ
    Inhib_Prop_c = param[6]

    # Henry's constants calculated for given temperature: Calculation was done with Van 't Hoff extrapolation to intorduce temperature correction of coefficient
    # H(T1) ≈ H(T0) * e ^ ((ΔH/R) * (1/T1 - 1/T0)) is the formula. ΔH is the enthalphy of the process, and H(T) Henry's constant
    # ⚠️Warning! this formula is applicable for not big temperature changes from standard conditions.
    K_H_h2 = 7.8 * 10 ** -4 * numpy.exp(-4180 / (100 * initial_parameter['R']) * (
                1 / initial_parameter['Temperature_0'] - 1 / initial_parameter[
            'Temperature_ex']))  # Mliq.bar^-1 #7.38*10^-4
    K_H_CH4 = 0.0014 * numpy.exp((-14240 / (100 * initial_parameter['R'])) * (
                1 / initial_parameter['Temperature_0'] - 1 / initial_parameter[
            'Temperature_ex']))  # Mliq.bar^-1 #0.00116
    K_H_co2 = 0.035 * numpy.exp((-19410 / (100 * initial_parameter['R'])) * (
                1 / initial_parameter['Temperature_0'] - 1 / initial_parameter[
            'Temperature_ex']))  # Mliq.bar^-1 #0.0271

    # calculate stoichiometric parameters from dynamically from dynamic growth yield
    stoech_Prop_c = stoech_Prop(initial_parameter['Growth_yield_Prop'])
    stoech_H2_c = stoech_H2(initial_parameter['Growth_yield_H2'])

    v_CO2_h = stoech_H2_c['CO2'] + stoech_H2_c['HCO3']
    v_CH4_h = stoech_H2_c['CH4']
    v_XH2 = stoech_H2_c['Biomass']
    
    v_CO2_Prop = stoech_Prop_c['CO2'] + stoech_Prop_c['HCO3']
    v_H2_Prop = stoech_Prop_c['H2']
    v_X_Prop = stoech_Prop_c['Biomass']
    v_Ac_Prop = stoech_Prop_c['AC']

    def model(state_zero, t):

        C_H2_M = state_zero[Id['C_H2_M']]  # initial hydrogen concentration in liquid in mol/l
        X_H2_gX = state_zero[Id['X_H2_gX']]  # initial biomass concentration of hydrogenotrophs in gX/mol(hydrogen)
        X_Prop_gX = state_zero[Id['X_Prop_gX']]  # initial biomass concentration of acetate oxidizers in gX/mol(acetate)
        C_CO2_M = state_zero[Id['C_CO2_M']]  # initial concentration of CO2 in the liquid mol/l
        C_propionate_M = state_zero[Id['C_propionate_M']]  # initial concentration of acetate mol/l
        C_CH4_M = state_zero[Id['C_CH4_M']]  # initial concentraiton of CH4 in the liquid mol/l
        C_H2_gas_M = state_zero[Id['C_H2_gas_M']]  # hydrogen gas concentration in the headspace mol/l
        C_CH4_gas_M = state_zero[Id['C_CH4_gas_M']]  # CH4 gas concentration in the headspace mol/l
        C_CO2_gas_M = state_zero[Id['C_CO2_gas_M']]  # CO2 gas concentration in the headspace mol/l
        Dead_Biomass = state_zero[Id['Dead_Biomass']]# initial total dead biomass is assumed to be zero
        C_acetate_M = state_zero[Id['C_acetate_M']]
        # estimation of N2 partial pressure in the headspace.
        # As headspace was flushed with N2, then the difference between sum of main gases and total pressure with help to estimate N2 partial pressure
        #        PP_gas_N2 = initial_parameter['inititla_total_pressure'] - (initial_parameter['CO2_patial_pressure'] + initial_parameter['H2_partial_pressure'] + initial_parameter['CH4_partial_pressure'] + initial_parameter['CO_partial_pressure']) # total head space total pressure at starting point is 1.72 bar: 80% N2 and 20% CO2

        # Inhibition function of propionate oxidizers and hydrogenotrophic methanogens
        Inhib_Prop = 1 / (1 + C_H2_M / Inhib_Prop_c)

        # Uptake rate equations
        rho_H2 = (qs_max_H2 * C_H2_M / (C_H2_M + Ks_H2)) * X_H2_gX
        rho_Prop = (qs_max_Prop * C_propionate_M / (C_propionate_M + Ks_Prop)) * X_Prop_gX * Inhib_Prop

        # Cell death rate equations
        rho_d_H2 = 8.33 * 1e-4 * X_H2_gX  # (1/(1+numpy.exp(-dG_cat_h2 - 11))) * X_H2_gX
        rho_d_Prop = 8.33 * 1e-4 * X_Prop_gX  # (1/(1+numpy.exp(-dG_cat_Ac - 0))) * X_Ac_gX

        # partial pressure of the gas calculation in Bars
        PP_gas_H2 = (C_H2_gas_M * initial_parameter['R'] * initial_parameter['Temperature_ex'])
        PP_gas_CH4 = (C_CH4_gas_M * initial_parameter['R'] * initial_parameter['Temperature_ex'])
        PP_gas_CO2 = (C_CO2_gas_M * initial_parameter['R'] * initial_parameter['Temperature_ex'])

        # gas-liquid mass transfer rate, based on Henry's law. 16 and 64 are COD equivalents of the H2 and CH4 respectivly
        Rho_T_H2 = (k_L_a * (C_H2_M - K_H_h2 * PP_gas_H2))
        Rho_T_CH4 = (k_L_a * (C_CH4_M - K_H_CH4 * PP_gas_CH4))
        Rho_T_CO2 = (k_L_a * ((C_CO2_M) - K_H_co2 * PP_gas_CO2))

        # Differential equation
        dH2dt = - rho_H2 + v_H2_Prop * rho_Prop - Rho_T_H2  # H2 concentration change

        dPropdt = - rho_Prop

        dAcdt = rho_Prop * v_Ac_Prop

        dXH2dt = v_XH2 * rho_H2 - rho_d_H2  # hydrogenotroph biomass change

        dXPropdt = v_X_Prop * rho_Prop - rho_d_Prop  # acetate oxidizer biomass change

        dCO2dt = (v_CO2_Prop * rho_Prop - v_CO2_h * rho_H2 - Rho_T_CO2)  # CO2 concentration change in the liquid. Beside gas-liquid mass transfer there is also dissociation to bicabonate

        dCH4dt = v_CH4_h * rho_H2 - Rho_T_CH4  # methane concentration change in the liquid

        dDdt = rho_d_Prop + rho_d_H2  # total dead biomass for calculation of mass balance


        dH2dt_gas = (Rho_T_H2 * initial_parameter['reaction_media_volume'] / initial_parameter['head_space'])
        dCH4dt_gas = (Rho_T_CH4 * initial_parameter['reaction_media_volume'] / initial_parameter['head_space'])
        dCO2dt_gas = (Rho_T_CO2 * initial_parameter['reaction_media_volume'] / initial_parameter['head_space'])

        return dH2dt, dXH2dt, dXPropdt, dCO2dt, dPropdt, dCH4dt, dH2dt_gas, dCH4dt_gas, dCO2dt_gas, dDdt, dAcdt

    # time range and time step of the integration of differential equaitons
    time_series = numpy.linspace(0, initial_parameter['duration_of_experiment'],
                                  int(initial_parameter['duration_of_experiment'] / initial_parameter['time_step']))

    starting_values = dict(C_H2_M=initial_parameter['H2_partial_pressure'] * K_H_h2,
                            # hydrogen initial concentration calcualted with Henry's law
                            X_H2_gX=param[0],
                            X_Prop_gX=param[1],
                            C_CO2_M=initial_parameter['CO2_patial_pressure'] * K_H_co2,
                            C_propionate_M=initial_parameter['Propionate_M'],
                            C_CH4_M=initial_parameter['CH4_partial_pressure'] * K_H_CH4,
                            C_H2_gas_M=initial_parameter['H2_partial_pressure'] / (
                                        initial_parameter['R'] * initial_parameter['Temperature_ex']),
                            # calculate concentration of gases from bar -> mol/l assuming gases behave as ideal gases
                            C_CH4_gas_M=initial_parameter['CH4_partial_pressure'] / (
                                        initial_parameter['R'] * initial_parameter['Temperature_ex']),
                            C_CO2_gas_M=initial_parameter['CO2_patial_pressure'] / (
                                        initial_parameter['R'] * initial_parameter['Temperature_ex']),
                            Dead_Biomass = 0,
                            C_acetate_M = initial_parameter['Acetate_M'])
  
    # solve differential equation
    starting_values_array = np.array([starting_values[k] for k in State_index])
    solution = odeint(model, starting_values_array, time_series)  # with parameter tcrit critical point is set, where integration care should be taken. This is the time point where gas was released from headspace.

    # produce output of whole model in pandas dataframe for ease of use later
    output = dict(Hydrogen=[], Biomass_HO=[], Biomass_PO=[], CO2=[], Propionate=[], CH4=[],
                  Hydrogen_gas=[], CH4_gas=[], CO2_gas=[], Dead_biomass=[], Acetate=[])
    for x in solution:
        output['Hydrogen'].append(x[0])
        output['Biomass_HO'].append(x[1])
        output['Biomass_PO'].append(x[2])
        output['CO2'].append(x[3])
        output['Propionate'].append(x[4])
        output['CH4'].append(x[5])
        output['Hydrogen_gas'].append(x[6])
        output['CH4_gas'].append(x[7])
        output['CO2_gas'].append(x[8])
        output['Dead_biomass'].append(x[9])
        output['Acetate'].append(x[10])

    return pd.DataFrame(output)


def plotter(H2_experiment, Ac_experiment, Prop_experiment, CH4_experiment, time, opt_simulation):
  time_series = numpy.linspace(0, 630, 12600)
  fig = make_subplots(rows=1, cols=1, subplot_titles=("Line Chart with Dropdown Menu"))

  # make plot of H2, CO2, CH4, propionate, and acetate with dropdown menu.
  fig.update_layout(updatemenus=[dict(buttons=[dict(args=[{'y': [opt_simulation['Hydrogen_gas'], H2_experiment], 'name': ['Hydrogen model', 'Hydrogen experiment']}],
                                                        label='H2 gas',
                                                        method='restyle'),
                                                dict(args=[{'y': [opt_simulation['Acetate'] , Ac_experiment], 'name': ['Acetate model', 'Acetate experiment']}],
                                                    label='Acetate',
                                                    method='restyle'),
                                                dict(args=[{'y': [opt_simulation['Propionate'], Prop_experiment], 'name': ['Propionate model', 'Propionate experiment']}],
                                                    label='Propionate',
                                                    method='restyle'),
                                              dict(args=[{'y': [opt_simulation['CH4_gas'], CH4_experiment], 'name': ['CH4 model', 'CH4 experiment']}],
                                                    label='CH4',
                                                    method='restyle')],
                                        direction='down',
                                        showactive=True,
                                        x=0.1,
                                        xanchor='left',
                                        y=1.15,
                                        yanchor='top')])

  # Add traces for initial y-axis columns
  fig.add_trace(go.Scatter(x=time_series / 24, y=opt_simulation['Hydrogen_gas'], mode='lines', name='Hydrogen model', line=dict(width=6)))
  fig.add_trace(go.Scatter(x=time, y=H2_experiment, mode='markers', name='Hydrogen experiment', connectgaps=True, marker=dict(size=13)))


  # Update layout
  fig.update_layout(xaxis=dict(title='Time (days)'), yaxis=dict(title='Concentration (mol/l)'))
  fig.update_layout(xaxis=dict(title='Time(days)', tickfont=dict(size=27)),
                    yaxis=dict(title='Concentration (mol/l)', tickfont=dict(size=27), tickformat='.1e'),
                    font=dict(size=20))

  return fig

def plotter_CSTR(opt_simulation, H2_steady_state, Propionate_steady_state):
  time_series = numpy.linspace(0, 4800, 96000)
  # make plot of H2, CO2, CH4, propionate, and acetate with dropdown menu.
  fig = make_subplots(rows=1, cols=1, subplot_titles=("Line Chart with Dropdown Menu"))
  fig.update_layout(updatemenus=[dict(buttons=[dict(args=[{'y': [opt_simulation['Hydrogen_gas'], [H2_steady_state] * (len(time_series)//24)], 'name': ['Hydrogen model', 'Hydrogen experiment']}],
                                                        label='H2 gas',
                                                        method='restyle'),
                                                dict(args=[{'y': [opt_simulation['Propionate'], [Propionate_steady_state] * (len(time_series)//24)], 'name': ['Propionate model', 'Propionate experiment']}],
                                                    label='Propionate',
                                                    method='restyle')],
                                        direction='down',
                                        showactive=True,
                                        x=0.1,
                                        xanchor='left',
                                        y=1.15,
                                        yanchor='top')])

  # Add traces for initial y-axis columns
  fig.add_trace(go.Scatter(x=time_series / 24, y=opt_simulation['Hydrogen_gas'], mode='lines', name='Hydrogen model', line=dict(width=6)))
  fig.add_trace(go.Scatter(x=time_series, y=[H2_steady_state] * (len(time_series)//24), mode="lines", name="Threshold", line=dict(dash="dash", color="red")))


  # Update layout
  fig.update_layout(xaxis=dict(title='Time (days)'), yaxis=dict(title='Concentration (mol/l)'))
  fig.update_layout(xaxis=dict(title='Time(days)', tickfont=dict(size=27)),
                    yaxis=dict(title='Concentration (mol/l)', tickfont=dict(size=27), tickformat='.1e'),
                    font=dict(size=20))
  return fig
