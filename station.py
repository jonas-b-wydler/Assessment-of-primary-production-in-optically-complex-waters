import pandas as pd
import numpy as np
from utility import calculate_pb_opt, linear_interpolate
from scipy.stats import pearsonr
import inspect


class Station:
    def __init__(self, station_id, lake, latitude, longitude, date, **kwargs):
        self.station_id = station_id
        self.lake = lake
        self.latitude = latitude
        self.longitude = longitude
        self.date = date
        self.models = []
        # Initialize the properties
        self._is_outlier       = False     # boolean flag
        self.outlier_distance  = None      # numeric score (e.g. PCA distance)
        self.outlier_issues    = []        # list of messages
        # -----------------------------------------
        
        # add a season
        self.month = self.date.month
        self.season = (
            'winter' if self.month in (12, 1, 2)
            else 'spring' if self.month in (3, 4, 5)
            else 'summer' if self.month in (6, 7, 8)
            else 'autumn'
        )
        # Define variables with dimensions
        self.variables = {
            'pp_hourly_depth_resolved': {'value': kwargs.get('pp_hourly_depth_resolved'), 'dims': ('depth',)},
            'pp_hourly_depth_integrated': {'value': kwargs.get('pp_hourly_depth_integrated'), 'dims': ()},
            'pp_daily_depth_integrated': {'value': kwargs.get('pp_daily_depth_integrated'), 'dims': ()},
            'chl_a_depth_resolved': {'value': kwargs.get('chl_a_depth_resolved'), 'dims': ('depth',)},
            'chl_a_depth_integrated': {'value': kwargs.get('chl_a_depth_integrated'), 'dims': ()},
            'chl_a_surface': {'value': kwargs.get('chl_a_surface'), 'dims': ()},
            'a_phy_surface': {'value': kwargs.get('a_phy'), 'dims': ('wavelength',)},
            'a_phy_depth_integrated': {'value': kwargs.get('a_phy'), 'dims': ('wavelength',)},
            'a_phy_depth_resolved': {'value': kwargs.get('a_phy'), 'dims': ('depth','wavelength')},
            'a_depth_integrated': {'value': kwargs.get('a'), 'dims': ('wavelength',)},
            'surface_temperature': {'value': kwargs.get('surface_temperature'), 'dims': ()},
            'temperature_depth_resolved': {'value': kwargs.get('temperature_depth_resolved'), 'dims': ('depth',)},
            'secchi': {'value': kwargs.get('secchi'), 'dims': ()},
            'par': {'value': kwargs.get('par'), 'dims': ()},
            'par_max': {'value': kwargs.get('par_max'), 'dims': ()},
            'par_noon': {'value': kwargs.get('par_noon'), 'dims': ('wavelength',)},
            'par_wavelength': {'value': kwargs.get('par_wavelength'), 'dims': ('wavelength',)},
            'par_hourly': {'value': kwargs.get('par_hourly'), 'dims': ('time',)},
            'par_hourly_wavelength': {'value': kwargs.get('par_hourly_wavelength'), 'dims': ('time','wavelength')},
            'kd': {'value': kwargs.get('kd'), 'dims': ()},
            'kds': {'value': kwargs.get('kds'), 'dims': ('wavelength',)},
            'kd_par': {'value': kwargs.get('kd_par'), 'dims': ()},
            'zeu': {'value': kwargs.get('zeu'), 'dims': ()},
            'dl': {'value': kwargs.get('dl'), 'dims': ()},
            'pb_opt': {'value': kwargs.get('pb_opt'), 'dims': ('depth',)},
            'surface_pb_opt': {'value': kwargs.get('surface_pb_opt'), 'dims': ()},
            'surface_pb_opt_measured': {'value': kwargs.get('surface_pb_opt_measured'), 'dims': ()},
            'an': {'value': kwargs.get('an'), 'dims': ('depth',)},
            'surface_an': {'value': kwargs.get('surface_an'), 'dims': ()},
            'chla_and_a_phy_depths': {'value': kwargs.get('chla_and_a_phy_depths'), 'dims': ('depth',)},
            'pp_measurement_depths': {'value': kwargs.get('pp_measurement_depths'), 'dims': ('depth',)},
            'incubation_start': {'value': kwargs.get('incubation_start'), 'dims': ()},
            'incubation_end': {'value': kwargs.get('incubation_end'), 'dims': ()},
            'incubation_time': {'value': kwargs.get('incubation_time'), 'dims': ()},
            'incubation_hours': {'value': kwargs.get('incubation_hours'), 'dims': ('time',)},
            'pp_peak_depth': {'value': None, 'dims': ()}

        }
    

    def add_model(self, model):
        self.models.append(model)
    
    @property
    def is_outlier(self):
        return self._is_outlier

    @is_outlier.setter
    def is_outlier(self, value: bool):
        self._is_outlier = bool(value)

    def compute_pp_peak_depth(self):
        pp_profile = self.variables['pp_hourly_depth_resolved']['value']
        depths = pp_profile.index.to_numpy(float)
        values = pp_profile.to_numpy(float).flatten()
    
        idx = np.nanargmax(values)
        peak_depth = float(depths[idx])
    
        self.variables['pp_peak_depth']['value'] = peak_depth

    def trapezoidal_integration_PP(self):
        pp_hourly_depth_resolved = self.variables.get('pp_hourly_depth_resolved', {}).get('value', None)
        if pp_hourly_depth_resolved is None:
            self.variables['pp_hourly_depth_integrated']['value'] = np.nan
            return

        depths = pp_hourly_depth_resolved.index.values
        values = np.array(pp_hourly_depth_resolved.values.flatten(), dtype=np.float64)
        nan_mask = np.isnan(values)

        if np.sum(nan_mask) > 5:
            self.variables['pp_hourly_depth_integrated']['value'] = np.nan
            return

        if nan_mask.any() and len(depths[~nan_mask]) > 0:
            values = np.interp(depths, depths[~nan_mask], values[~nan_mask])

        self.variables['pp_hourly_depth_integrated']['value'] = np.trapz(values, depths)

    def interpolate_chla_pp(self):
        # Retrieve existing DataFrame or Series for chl_a and PP
        chl_a_data = self.variables['chl_a_depth_resolved']['value']
        pp_hourly_data = self.variables['pp_hourly_depth_resolved']['value']
    
        chl_a_values = pd.to_numeric(chl_a_data.values.flatten(), errors='coerce')
        pp_hourly_values = pd.to_numeric(pp_hourly_data.values.flatten(), errors='coerce')

        interpolated_chl_a = linear_interpolate(chl_a_data.index.values, chl_a_values)
        interpolated_pp_hourly = linear_interpolate(pp_hourly_data.index.values, pp_hourly_values)

        self.variables['chl_a_depth_resolved']['value'] = pd.DataFrame(
            interpolated_chl_a, index=chl_a_data.index, columns=['chl_a_depth_resolved']
        )
        self.variables['pp_hourly_depth_resolved']['value'] = pd.DataFrame(
            interpolated_pp_hourly, index=pp_hourly_data.index, columns=['pp_hourly_depth_resolved']
        )
                 
    def a_phy(self):
        # Define the wavelength range (10 nm resolution)
        wv = np.arange(400, 710, 10)
    
        # Interpolated As and Bs
        intercept = np.array([0.033, 0.03840468, 0.045, 0.05008251, 0.053, 0.0509849,
                              0.048, 0.04547831, 0.042, 0.03696949, 0.031, 0.02498517,
                              0.019, 0.01539597, 0.013, 0.01045871, 0.009, 0.00811671,
                              0.007, 0.00797494, 0.008, 0.00802645, 0.009, 0.00949961,
                              0.01, 0.01138279, 0.014, 0.02161268, 0.023, 0.0129098,
                              0.005])
        slope = np.array([0.233, 0.24552237, 0.259, 0.27323491, 0.281, 0.29265144,
                          0.299, 0.29762277, 0.307, 0.29821063, 0.258, 0.21450849,
                          0.172, 0.13739559, 0.111, 0.09845876, 0.103, 0.11762342,
                          0.134, 0.1619905, 0.184, 0.17937802, 0.164, 0.17907438,
                          0.193, 0.211646, 0.219, 0.19884796, 0.159, 0.11693024,
                          0.137])
    
        # Depth-integrated chlorophyll-a
        if self.variables['chl_a_depth_integrated']['value'] is not None:
            chl_a_val = self.variables['chl_a_depth_integrated']['value']
            if chl_a_val == 0:
                a_phy_val = 0
            else:
                a_phy_val = intercept * (chl_a_val ** (-slope)) * chl_a_val
            self.variables['a_phy_depth_integrated']['value'] = a_phy_val
    
        # Surface chlorophyll-a
        if self.variables['chl_a_surface']['value'] is not None:
            chl_a_val = self.variables['chl_a_surface']['value']
            if chl_a_val == 0:
                a_phy_val = 0
            else:
                a_phy_val = intercept * (chl_a_val ** (-slope)) * chl_a_val
            self.variables['a_phy_surface']['value'] = a_phy_val
    
        # Depth-resolved chlorophyll-a
        if self.variables['chl_a_depth_resolved']['value'] is not None:
            chl_a = self.variables['chl_a_depth_resolved']['value'].values.astype(float)
    
            # Shape chl_a into (depth, 1)
            chl_a = chl_a.reshape(-1, 1)
            # Repeat chl_a across the wavelength dimension to match intercept/slope
            chl_a = np.repeat(chl_a, len(intercept), axis=1)  # (depth, wavelength)
    
            # Initialize a_phy array
            a_phy = np.zeros_like(chl_a, dtype=float)
    
            # Create a mask for non-zero values
            nonzero_mask = (chl_a != 0)
    
            # Compute a_phy for all depths and wavelengths
            a_phy_full = intercept * (chl_a ** (-slope)) * chl_a
    
            # Assign only where chl_a is non-zero
            a_phy[nonzero_mask] = a_phy_full[nonzero_mask]
    
            self.variables['a_phy_depth_resolved']['value'] = a_phy


    def par_wavelengths(self):
        
        PARfraction = np.array([0.00227, 0.00218, 0.00239, 0.00189, 0.00297, 0.00348, 0.00345, 0.00344,
                            0.00373, 0.00377, 0.00362, 0.00364, 0.00360, 0.00367, 0.00354, 0.00368, 
                            0.00354, 0.00357, 0.00363, 0.00332, 0.00358, 0.00357, 0.00359, 0.00340, 
                            0.00350, 0.00332, 0.00342, 0.00347, 0.00342, 0.00290, 0.00314])
        
        if self.variables['par']['value'] is not None:
            par = self.variables['par']['value']
            par_noon = np.pi * par / 2 * PARfraction
            par_wavelength = np.pi * par / 2  * PARfraction
            self.variables['par_noon'] = {'value': par_noon, 'dims': ('wavelength',)}
            self.variables['par_wavelength'] = {'value': par_wavelength, 'dims': ('wavelength',)}
            
        if self.variables['par_hourly']['value'] is not None:
            par_wavelength_resolved = np.zeros((24, 31))
            par_hourly = self.variables['par_hourly']['value'] 
            for t in range(len(par_hourly)):
                par_wavelength_resolved[t, :] = par_hourly[t] * PARfraction # * np.pi /2           
             
            self.variables['par_hourly_wavelength'] = {'value': par_wavelength_resolved, 'dims': ('time', 'wavelength',)}
            
        

    def kd_from_secchi(self):
        PARfraction = np.array([0.00227, 0.00218, 0.00239, 0.00189, 0.00297, 0.00348, 0.00345, 0.00344,
                            0.00373, 0.00377, 0.00362, 0.00364, 0.00360, 0.00367, 0.00354, 0.00368, 
                            0.00354, 0.00357, 0.00363, 0.00332, 0.00358, 0.00357, 0.00359, 0.00340, 
                            0.00350, 0.00332, 0.00342, 0.00347, 0.00342, 0.00290, 0.00314])
        
        if self.variables['secchi']['value'] is not None:
            if self.variables['secchi']['value'] == 0:
                self.variables['secchi']['value'] = 0.01
                    
            kd = 2.2 / self.variables['secchi']['value']
            kds = np.full_like(PARfraction, kd)
            kd_par = kd
            self.variables['kd'] = {'value': kd, 'dims': ()}
            self.variables['kds'] = {'value': kds, 'dims': ('wavelength',)}
            self.variables['kd_par'] = {'value': kd_par, 'dims': ()}

    def zeu(self):
        if (self.variables['secchi']['value'] is not None
                and self.variables['par_max']['value'] is not None
                and self.variables['par_max']['value'] > 0):
            kd_par = self.variables['kd_par']['value']
            zeu = -1 * np.log(0.01) / kd_par
            self.variables['zeu'] = {'value': zeu if zeu > 0 else np.nan, 'dims': ()}

    def set_PbOpt(self):
        depth_resolved_temperature = self.variables.get('temperature_depth_resolved', {}).get('value')
        surface_temperature = self.variables.get('surface_temperature', {}).get('value')

        if depth_resolved_temperature is not None:
            depth_resolved_temperature = np.array(depth_resolved_temperature, dtype=float).flatten()

            if np.isnan(depth_resolved_temperature).all():
                self.variables['pb_opt'] = {'value': np.nan * np.ones_like(depth_resolved_temperature), 'dims': ('depth',)}
                self.variables['surface_pb_opt'] = {'value': np.nan, 'dims': ()}
            else:
                pb_opt_values = np.array(
                    [calculate_pb_opt(temp) if not np.isnan(temp) else np.nan for temp in depth_resolved_temperature],
                    dtype=float
                )
                self.variables['pb_opt'] = {'value': pb_opt_values, 'dims': ('depth',)}

                top_three_depths = pb_opt_values[:3]
                if not np.isnan(top_three_depths).all():
                    surface_pb_opt = np.nanmean(top_three_depths)
                else:
                    surface_pb_opt = np.nan
                
                self.variables['surface_pb_opt'] = {'value': surface_pb_opt, 'dims': ()}
    
        elif surface_temperature is not None:
            surface_temperature = float(surface_temperature)
            surface_pb_opt = np.nan if np.isnan(surface_temperature) else calculate_pb_opt(surface_temperature)
            self.variables['surface_pb_opt'] = {'value': surface_pb_opt, 'dims': ()}
            self.variables['pb_opt'] = {'value': np.nan * np.ones(1), 'dims': ('depth',)}
        else:
            self.variables['pb_opt'] = {'value': np.nan * np.ones(1), 'dims': ('depth',)}
            self.variables['surface_pb_opt'] = {'value': np.nan, 'dims': ()}
    
    def set_surface_temp(self):
        # Return early if a valid surface temperature already exists
        existing_temp = self.variables.get('surface_temperature', {}).get('value')
        try:
            if np.isfinite(float(existing_temp)):
                return
        except (ValueError, TypeError):
            pass

        # Otherwise, calculate from depth profile
        surface_temp = np.nan
        temp_profile = self.variables.get('temperature_depth_resolved', {}).get('value')
        if temp_profile is not None:
            target_series = temp_profile.iloc[:, 0] if isinstance(temp_profile, pd.DataFrame) else temp_profile
            try:
                surface_temp = target_series.head(3).mean()
            except AttributeError:
                surface_temp = np.nan

        self.variables['surface_temperature'] = {'value': surface_temp, 'dims': ()}
  
    def set_surface_chl_a(self):
        chl_a_depth_resolved = self.variables.get('chl_a_depth_resolved', {}).get('value')

        if chl_a_depth_resolved is not None and len(chl_a_depth_resolved) >= 3:
            chl_a_depth_resolved = np.array(chl_a_depth_resolved)
            top_four_depths = chl_a_depth_resolved[:3]
            surface_chl_a = np.nanmean(top_four_depths)
            self.variables['chl_a_surface'] = {'value': surface_chl_a, 'dims': ()}
        else:
            if 'chl_a_surface' not in self.variables:
                self.variables['chl_a_surface'] = {'value': np.nan, 'dims': ()}

    def set_depths(self):
        chl_a_depths = None
        chl_a_var = self.variables.get('chl_a_depth_resolved', {}).get('value')
        if chl_a_var is not None:
            chl_a_depths = chl_a_var.index.values
            if pd.isna(chl_a_depths).any():
                chl_a_depths = None

        pp_depths = None
        pp_var = self.variables.get('pp_hourly_depth_resolved', {}).get('value')
        if pp_var is not None:
            pp_depths = pp_var.index.values
            if pd.isna(pp_depths).any():
                pp_depths = None

        self.variables['chla_and_a_phy_depths']['value'] = chl_a_depths
        self.variables['pp_measurement_depths']['value'] = pp_depths


    @staticmethod
    def _parse_hour(time_str):
        """Extract the hour integer from a 'HH:MM' string, or return None."""
        if time_str is None:
            return None
        try:
            return int(time_str.split(':')[0])
        except (ValueError, AttributeError):
            return None

    def set_incubation_hours(self):
        start_hour = self._parse_hour(self.variables['incubation_start']['value'])
        incubation_duration = self.variables['incubation_time']['value']

        if start_hour is None:
            self.variables['incubation_hours']['value'] = None
            return

        # If duration is missing, try to compute from start and end times
        if incubation_duration is None or (isinstance(incubation_duration, float) and np.isnan(incubation_duration)):
            end_hour = self._parse_hour(self.variables['incubation_end']['value'])
            if end_hour is None:
                self.variables['incubation_hours']['value'] = None
                return
            incubation_duration = end_hour - start_hour
            # If negative, assume it wraps past midnight
            if incubation_duration <= 0:
                incubation_duration += 24
        else:
            incubation_duration = int(round(incubation_duration))

        if incubation_duration > 0:
            self.variables['incubation_hours']['value'] = np.array(
                [(start_hour + i) % 24 for i in range(incubation_duration)]
            )
        else:
            self.variables['incubation_hours']['value'] = None

    def set_daily_pp(self):
        hourly_value = self.variables['pp_hourly_depth_integrated']['value']
        daily_value = self.variables['pp_daily_depth_integrated']['value']
        day_length = self.variables['dl']['value']
    
        # If hourly value is available, calculate daily value if it's missing
        if hourly_value is not None and not np.isnan(hourly_value) and (daily_value is None or np.isnan(daily_value)):
            self.variables['pp_daily_depth_integrated']['value'] = hourly_value * day_length
        
        # If daily value is available, calculate hourly value if it's missing
        elif daily_value is not None and not np.isnan(daily_value) and (hourly_value is None or np.isnan(hourly_value)):
            self.variables['pp_hourly_depth_integrated']['value'] = daily_value / day_length
        
    def apply_model(self, model):
        # Match model's calculate_pp signature to station variables
        sig = inspect.signature(model.calculate_pp)
        model_args = {param: self.variables[param]['value'] for param in sig.parameters.keys()}
        model.calculate_pp(**model_args)
        self.add_model(model)

   


    def calculate_model_error(self):
        # Observed integrated PP
        measured_pp = self.variables['pp_hourly_depth_integrated']['value']

        # Observed vertical profile (depth-resolved)
        obs_profile_series = self.variables.get('pp_hourly_depth_resolved', {}).get('value')
        obs_depths = obs_profile_series.index.values.astype(float)
        obs_values = np.squeeze(obs_profile_series.values.astype(float))

        dl = self.variables['dl']['value']

        for model in self.models:
            if not hasattr(model, 'model_pp_z'):
                model.mae = None
                model.mape = None
                model.pattern_correlation = None
                continue

            model_depths = model.depths
            # Convert daily PP per depth to hourly
            model_pp_z = model.model_pp_z / dl

            # Integrate over depth and compare to measured PP => MAE, MAPE
            residual = measured_pp - np.trapz(model_pp_z, model_depths)
            model.mae = abs(residual)
            model.mape = abs(residual / measured_pp) * 100 if measured_pp != 0 else None

            # Pattern correlation (shape-only):
            # Align the model's values to the observed depths (nearest depth)
            aligned_model_values = np.array(
                [model_pp_z[np.abs(model_depths - d).argmin()] for d in obs_depths]
            )

            # Integrate both profiles over the observed depths
            obs_integral = np.trapz(obs_values, obs_depths)
            mod_integral = np.trapz(aligned_model_values, obs_depths)

            if obs_integral == 0.0 or mod_integral == 0.0:
                model.pattern_correlation = None
            else:
                # Normalize each profile by its integral and compute Pearson correlation
                pcorr_shape, _ = pearsonr(obs_values / obs_integral, aligned_model_values / mod_integral)
                model.pattern_correlation = pcorr_shape