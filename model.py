import numpy as np
from utility import integrate_wavelengths, integrate_hours, integrate_depth
from scipy.interpolate import PchipInterpolator
import pandas as pd

class BaseModel:
    def __init__(self, name):
        self.name = name
        self.model_pp = 0
        self.hour = np.arange(0, 24)
        self.residual = None
        self.mae = None
        self.mape = None
        self.pattern_correlation = None

class depth_resolved_pp_model(BaseModel):
    def __init__(self, zeu=10, zlim = 30, name='depth_resolved_pp_model',correction_method = 'A', vert_uni = False, **kwargs):
        super().__init__(name=name)  # Pass the name parameter to the base class
        self.delz = 0.1
        self.zlim = zlim
        self.zeu = zeu
        self.correction_method = correction_method
        self.vert_uni = vert_uni
        self.depths = np.linspace(0, self.zlim, int(self.zlim / self.delz) + 1)
        self.e_tz = np.zeros((len(self.depths), len(self.hour))) # irradiance
        self.ap_tz = np.zeros((len(self.depths), len(self.hour)))  # absorbed photons
        self.ap_z = np.zeros(len(self.depths))
        self.ap = 0
        self.eu = 1
        self.f_par = np.zeros((len(self.depths), len(self.hour)))
        self.model_pp_tz = np.zeros((len(self.depths), len(self.hour)))
        self.model_pp_z =  np.zeros(len(self.depths))
        self.model_pp = 0

    @property
    def is_depth_resolved(self):
        return True
    
    def interpolate_to_fixed_grid(self, input_depths, input_values):
        pchip_interp = PchipInterpolator(input_depths, input_values)
        interpolated_values = pchip_interp(self.depths)
        interpolated_values[interpolated_values < 0] = 0
        return interpolated_values
    
    def set_ap_tz(self, par_noon, par_hourly_wavelength, a_phy_depth_resolved, kds, chla_and_a_phy_depths, a_phy_surface):
        input_depths = chla_and_a_phy_depths
        input_values = a_phy_depth_resolved
        a_phy_interpolated = self.interpolate_to_fixed_grid(input_depths, input_values)

        for t in range(len(self.hour)):
            for z in range(len(self.depths)):
                self.e_tz[z, t] = integrate_wavelengths(par_hourly_wavelength[t,:] * 0.95 * np.exp(-kds * self.depths[z]))
                self.ap_tz[z, t] = integrate_wavelengths(par_hourly_wavelength[t,:] * 0.95 * np.exp(-kds * self.depths[z]) * a_phy_interpolated[z,:])

        self.ap_z = np.trapz(self.ap_tz, x=self.hour, axis=1)
        self.ap = np.trapz(self.ap_z, x=self.depths, axis=0)

    def set_upwelling_correction_A(self):

        self.eu = 1.4

        self.e_tz = self.e_tz *  self.eu
        self.ap_tz =  self.ap_tz * self.eu
        self.ap_z = self.ap_z * self.eu
        self.ap = self.ap * self.eu
    
        
    def set_upwelling_correction_B(self, chl_a_surface, kd_par):

        c_1 = 1.32 * (kd_par ** 0.153)                                               # Eq. 4: from Arst et al. (2012)
        c_2 = (0.0023 * chl_a_surface) + 0.016                                       # Eq. 5: from Arst et al. (2012)
        eu = c_1 * np.exp(c_2 * self.depths)                                         # Eq. 3: correction factor between scalar and planar PAR
        self.eu = eu
        
        self.e_tz = self.eu[:, np.newaxis] * self.e_tz 
        self.ap_tz = self.eu[:, np.newaxis] * self.ap_tz        
        self.ap_z = self.ap_z * self.eu
        self.ap = integrate_depth(self.ap_z, self.depths, self.zlim)
    
    def apply_upwelling_correction(self, chl_a_surface, kd_par, par_wavelength, a_phy_surface):
        if self.correction_method == 'A':
            self.set_upwelling_correction_A()
        elif self.correction_method == 'B':
            self.set_upwelling_correction_B(chl_a_surface, kd_par)
        else:
            raise ValueError(f"Unknown correction method: {self.correction_method}")


class lee_like_model(depth_resolved_pp_model):
    def __init__(self, zeu=10, name='lee_like_model', zlim=30, correction_method='A', vert_uni=False, **kwargs):
        super().__init__(zeu=zeu, zlim=zlim, correction_method=correction_method, vert_uni=vert_uni)
        self.name = name

    def quantum_yield_A(self, par_noon, par_hourly_wavelength, kd):
        f_par = 0.08 * ((0.4 * np.exp(-0.24 * self.e_tz)) / (0.4 + self.e_tz))
        self.f_par = f_par

    def calculate_pp(self, chl_a_surface, a_phy_surface, a_phy_depth_resolved, par_noon, par_hourly, par_hourly_wavelength, kd, kds, kd_par,
                     zeu, chla_and_a_phy_depths, pp_measurement_depths, par_wavelength):

        if self.vert_uni:
            a_phy_depth_resolved = np.full_like(a_phy_depth_resolved, a_phy_surface)
        self.pp_measurement_depths = pp_measurement_depths
        self.set_ap_tz(par_noon, par_hourly_wavelength, a_phy_depth_resolved, kds, chla_and_a_phy_depths, a_phy_surface)
        self.ap_tz = np.maximum(self.ap_tz, 0)
        self.e_tz = np.maximum(self.e_tz, 0) # set negative values to zero
        
        self.apply_upwelling_correction(chl_a_surface, kd_par, par_wavelength, a_phy_surface)
        self.quantum_yield_A(par_noon, par_hourly_wavelength, kd)

        self.model_pp_tz = self.f_par * self.ap_tz * 12000
        self.model_pp_tz = np.nan_to_num( self.model_pp_tz, nan=0)

        
        self.model_pp_t = np.apply_along_axis(integrate_depth, 0, self.model_pp_tz, self.depths, z_lim=self.zlim)
        self.model_pp_z = np.apply_along_axis(integrate_hours, 1, self.model_pp_tz)
        self.model_pp = integrate_depth(self.model_pp_z, self.depths, self.zlim)


class arst_like_model(depth_resolved_pp_model):
    def __init__(self, zeu=10, name='arst_like_model', zlim=30, correction_method='A', vert_uni=False, **kwargs):
        super().__init__(zeu=zeu, zlim=zlim, correction_method=correction_method, vert_uni=vert_uni)
        self.name = name
        self.variables = {'m_depth_resolved': {'value': None, 'dims': ('depth', 'time')}}

    def calculate_M_depth_resolved_integral(self, chl_a_surface, chl_a_depth_resolved, chla_and_a_phy_depths, par_hourly_wavelength, kd_par):
        m_depth_resolved = np.zeros((len(self.depths), len(self.hour)))
        interpolated_values = self.interpolate_to_fixed_grid(chla_and_a_phy_depths, chl_a_depth_resolved)
        interpolated_values_2 = np.squeeze(interpolated_values)
        for t in range(len(self.hour)):
            for z in range(len(self.depths)):
                if interpolated_values_2[z] == 0:
                    m_depth_resolved[z,t] = 0
                else:
                    if interpolated_values_2[z] < 35:
                        m_depth_resolved[z,t] = 3.18 - (0.2125 * (kd_par ** 2.5)) + (0.34 * integrate_wavelengths(par_hourly_wavelength[t])* 0.95)
                    elif interpolated_values_2[z] < 80:
                        m_depth_resolved[z,t] = 3.58 - (0.31 * integrate_wavelengths(par_hourly_wavelength[t])* 0.95) - (0.0072 * interpolated_values_2[z])
                    elif interpolated_values_2[z] <= 120:
                        m_depth_resolved[z,t] = 2.46 - (0.106 * integrate_wavelengths(par_hourly_wavelength[t]) * 0.95) - (0.00083 * (interpolated_values_2[z] ** 1.5))
                    else:
                        m_depth_resolved[z,t] = 0.67
        
        # Set negative values of M to default
        m_depth_resolved[m_depth_resolved < 0] = 0.67
        
        self.variables['m_depth_resolved']['value'] = m_depth_resolved

    def quantum_yield_B(self, chl_a_surface, chl_a_depth_resolved, chla_and_a_phy_depths, par_noon, par_hourly_wavelength, kd_par):
        self.calculate_M_depth_resolved_integral(chl_a_surface, chl_a_depth_resolved, chla_and_a_phy_depths, par_hourly_wavelength, kd_par)
        f_par = 0.08 / ((1 + (self.variables['m_depth_resolved']['value'] * self.e_tz)) ** 1.5)
        self.f_par = f_par

    def calculate_pp(self, chl_a_surface, chl_a_depth_resolved, a_phy_surface, a_phy_depth_resolved, par_noon, par_hourly, par_hourly_wavelength, kd, kds, kd_par,
                     zeu, chla_and_a_phy_depths, pp_measurement_depths, par_wavelength):
        if self.vert_uni:
            a_phy_depth_resolved = np.full_like(a_phy_depth_resolved, a_phy_surface)
            chl_a_depth_resolved = np.full_like(chl_a_depth_resolved, chl_a_surface)

        self.pp_measurement_depths = pp_measurement_depths
        measurement_idx = (self.pp_measurement_depths / self.delz).astype(int)

        self.set_ap_tz(par_noon, par_hourly_wavelength, a_phy_depth_resolved, kds, chla_and_a_phy_depths, a_phy_surface)
        self.ap_tz = np.maximum(self.ap_tz, 0)
        self.e_tz = np.maximum(self.e_tz, 0)

        self.apply_upwelling_correction(chl_a_surface, kd_par, par_wavelength, a_phy_surface)
        self.quantum_yield_B(chl_a_surface, chl_a_depth_resolved, chla_and_a_phy_depths, par_noon, par_hourly_wavelength, kd_par)

        self.model_pp_tz = self.f_par * self.ap_tz * 12000
        self.model_pp_tz = np.nan_to_num(self.model_pp_tz, nan=0)

        self.model_pp_t = np.apply_along_axis(integrate_depth, 0, self.model_pp_tz, self.depths, z_lim=self.zlim)
        self.model_pp_z = np.apply_along_axis(integrate_hours, 1, self.model_pp_tz)
        self.model_pp = integrate_depth(self.model_pp_z, self.depths, self.zlim)

        self.pp_tz_meas = self.model_pp_tz[measurement_idx, :]


class cafe_like_model(depth_resolved_pp_model):
    def __init__(self, zeu=10, name='cafe_like_model', zlim=30, correction_method='A', vert_uni=False, **kwargs):
        super().__init__(zeu=zeu, zlim=zlim, correction_method=correction_method, vert_uni=vert_uni)
        self.name = name
        self.ek = np.empty(len(self.depths))
        self.phimax = 0
        self.kpur = 0
        self.e_tzw = np.zeros((len(self.depths), len(self.hour)))

    def quantum_yield_C(self, chla_and_a_phy_depths, a_phy_depth_resolved, a_phy_surface, par_noon, kd, kd_par, zeu, dl, par_hourly_wavelength, par):
        input_depths = chla_and_a_phy_depths
        input_values = a_phy_depth_resolved
        a_phy_interpolated = self.interpolate_to_fixed_grid(input_depths, input_values)

        self.ek = np.full(len(self.depths), 19 * np.exp(0.038 * (par * 0.95 / dl) ** 0.45 / kd_par))
        self.ek[self.ek < 0.036] = 0.036

        self.kpur = self.ek

        phirange = np.array([0.018, 0.08])
        ekrange = np.array([150 * 3600 / 1e6, 10 * 3600 / 1e6])

        slope = (phirange[1] - phirange[0]) / (ekrange[1] - ekrange[0])
        self.phimax = phirange[1] + (self.ek - ekrange[1]) * slope
        self.phimax[self.phimax < phirange[0]] = phirange[0]
        self.phimax[self.phimax > phirange[1]] = phirange[1]

    def calculate_pp(self, chl_a_surface, a_phy_depth_resolved, a_phy_surface, par_noon, par, par_hourly, par_wavelength, kd, kds, kd_par, zeu, dl, par_hourly_wavelength, chla_and_a_phy_depths):
        if self.vert_uni:
            a_phy_depth_resolved = np.full_like(a_phy_depth_resolved, a_phy_surface)

        self.set_ap_tz(par_noon, par_hourly_wavelength, a_phy_depth_resolved, kds, chla_and_a_phy_depths, a_phy_surface)
        self.ap_tz = np.maximum(self.ap_tz, 0)
        self.e_tz = np.maximum(self.e_tz, 0)

        self.apply_upwelling_correction(chl_a_surface, kd_par, par_wavelength, a_phy_surface)
        self.quantum_yield_C(chla_and_a_phy_depths, a_phy_depth_resolved, a_phy_surface, par_noon, kd, kd_par, zeu, dl, par_hourly_wavelength, par)

        self.kpur = self.kpur.reshape(-1, 1)
        self.phimax = self.phimax.reshape(-1, 1)
        self.f_par = self.phimax * np.tanh(self.kpur / self.e_tz)

        self.model_pp_tz = self.f_par * self.ap_tz * 12000
        self.model_pp_tz = np.nan_to_num(self.model_pp_tz, nan=0)

        self.model_pp_t = np.apply_along_axis(integrate_depth, 0, self.model_pp_tz, self.depths, z_lim=self.zlim)
        self.model_pp_z = np.apply_along_axis(integrate_hours, 1, self.model_pp_tz)
        self.model_pp = integrate_depth(self.model_pp_z, self.depths, self.zlim)




class vgpm_depth_resolved(depth_resolved_pp_model):
    def __init__(self, zeu=10, name='vgpm_depth_resolved', zlim=30, correction_method='A', vert_uni=False, **kwargs):
        super().__init__(zeu=zeu, zlim=zlim, correction_method=correction_method, vert_uni=vert_uni)
        self.name = name

    def calculate_pp(self, chl_a_surface, chl_a_depth_resolved, a_phy_surface, a_phy_depth_resolved, par_noon, par_hourly, par_hourly_wavelength, kd, kds, kd_par,
                     zeu, chla_and_a_phy_depths, pp_measurement_depths, par_wavelength, par, dl, surface_pb_opt):
        if self.vert_uni:
            chl_a_depth_resolved = np.full_like(chl_a_depth_resolved, chl_a_surface)

        self.pp_measurement_depths = pp_measurement_depths
        self.set_ap_tz(par_noon, par_hourly_wavelength, a_phy_depth_resolved, kds, chla_and_a_phy_depths, a_phy_surface)
        self.ap_tz = np.maximum(self.ap_tz, 0)
        self.e_tz = np.maximum(self.e_tz, 0)

        self.apply_upwelling_correction(chl_a_surface, kd_par, par_wavelength, a_phy_surface)

        self.ez = np.zeros(len(self.depths))
        self.ez = np.apply_along_axis(integrate_hours, 1, self.e_tz)
        self.emax = 0.3195 * 0.95 * par

        if par * 0.95 > 3:
            self.bd = -0.0203 * np.log(par * 0.95) + 0.124
        else:
            self.bd = 0.1

        x = par * 0.95
        optical_depth = (
            -7.56e-8 * x**4 +
            1.84e-5 * x**3 -
            1.71e-3 * x**2 +
            7.5e-2 * x -
            1.37e-3
        )

        self.eopt = par * np.exp(-optical_depth)

        interpolated_values = self.interpolate_to_fixed_grid(chla_and_a_phy_depths, chl_a_depth_resolved)

        self.model_pp_z = surface_pb_opt * np.squeeze(interpolated_values) * ((1 - np.exp(-self.ez / self.emax) * np.exp(-self.bd * self.ez)) / (1 - np.exp(-self.eopt / self.emax) * np.exp(-self.bd * self.eopt)))
        self.model_pp_z = self.model_pp_z * dl
        self.model_pp = integrate_depth(self.model_pp_z, self.depths, self.zlim)


class vgpm_model(BaseModel):
    def __init__(self, zeu=10, **kwargs):
        super().__init__('vgpm_model')
        self.model_pp = 0
        self.zeu = zeu

    @property
    def is_depth_resolved(self):
        return False

    def calculate_pp(self, par, surface_pb_opt_measured, surface_pb_opt, chl_a_surface, dl, zeu):
        if surface_pb_opt_measured is not None and not np.isnan(surface_pb_opt_measured):
            pb_opt = surface_pb_opt_measured
        else:
            pb_opt = surface_pb_opt

        self.model_pp = 0.66125 * pb_opt * (par * 0.95 / (par * 0.95 + 4.1)) * zeu * chl_a_surface * dl
