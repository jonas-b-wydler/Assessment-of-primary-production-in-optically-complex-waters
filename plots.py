import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import geopandas as gpd
from scipy import stats


class Plots:

    def plot_all_profiles_with_chla_and_zeu(
        self,
        all_lakes,
        variable_pp='pp_hourly_depth_resolved',
        variable_chla='chl_a_depth_resolved',
        variable_kd='kd_par',
        save_dir=None,
        fmt='svg'
    ):
        for lake_name, lake in all_lakes.items():
            print(f"Plotting modeled curve for lake: {lake_name}")

            lake.plot_all_profiles_with_chla_and_zeu(
                variable_pp=variable_pp,
                variable_chla=variable_chla,
                variable_kd=variable_kd,
                save_dir=save_dir,
                fmt=fmt
            )

    def plot_modeled_curve(self, all_lakes, variable_name='pp_hourly_depth_resolved'):
        for lake_name, lake in all_lakes.items():
            print(f"Plotting modeled curve for lake: {lake_name} using variable: {variable_name}")
            lake.plot_modeled_curve(variable_name)

    def plot_final_quadrant(self, df_to_plot, group_name='All Lakes', metric_y='MAPE', metric_x='p_corr', save_path=None):
        if df_to_plot.empty:
            return

        df_plot = df_to_plot.dropna(subset=[metric_x, metric_y]).copy()
        if df_plot.empty:
            return

        df_plot['chl_resolution'] = df_plot['uniform_chl'].apply(lambda x: 'Uniform' if x else 'Depth-Resolved')

        if 'group' in df_plot.columns:
            n_stations = df_plot[['group', 'station_date']].drop_duplicates().shape[0]
        elif 'lake_name' in df_plot.columns:
            n_stations = df_plot[['lake_name', 'station_date']].drop_duplicates().shape[0]
        else:
            n_stations = df_plot['station_date'].nunique()

        def q1(x): return x.quantile(0.25)
        def q3(x): return x.quantile(0.75)

        agg_df = df_plot.groupby(['model_full_name', 'base_model', 'correction', 'chl_resolution'])\
                               .agg({metric_x: ['median', q1, q3], metric_y: ['median', q1, q3]}).reset_index()
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

        agg_df = agg_df.sort_values(by='chl_resolution', ascending=False)

        x_lims = (0.25, 1.0)
        y_lims = (15, 200)

        x_median_line = agg_df[f'{metric_x}_median'].median()
        y_median_line = agg_df[f'{metric_y}_median'].median()

        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

        title_fontsize = 22
        label_fontsize = 16
        legend_fontsize = 12
        tick_fontsize = 12

        sns.scatterplot(
            data=agg_df, x=f'{metric_x}_median', y=f'{metric_y}_median',
            hue='base_model',
            style='chl_resolution',
            size='correction',
            sizes=(180, 450), ax=ax, palette='colorblind',
            edgecolor='black',
            linewidth=0.8,
            zorder=5
        )

        x_errors = [agg_df[f'{metric_x}_median'] - agg_df[f'{metric_x}_q1'], agg_df[f'{metric_x}_q3'] - agg_df[f'{metric_x}_median']]
        y_errors = [agg_df[f'{metric_y}_median'] - agg_df[f'{metric_y}_q1'], agg_df[f'{metric_y}_q3'] - agg_df[f'{metric_y}_median']]

        ax.errorbar(x=agg_df[f'{metric_x}_median'], y=agg_df[f'{metric_y}_median'], xerr=x_errors, yerr=y_errors,
                    fmt='none', ecolor='gray', elinewidth=0.8, capsize=2, alpha=0.6, zorder=1)

        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.axvline(x_median_line, color='black', linestyle='--', lw=1.5, zorder=0)
        ax.axhline(y_median_line, color='black', linestyle='--', lw=1.5, zorder=0)

        ax.set_title(f"Impact of Model Structure on Performance Metrics\n({group_name})", fontsize=title_fontsize, pad=20)
        ax.set_xlabel("Vertical Profile Shape Correlation (Pearson's r)", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Magnitude Error (Median MAPE, %)", fontsize=label_fontsize, labelpad=10)

        ax.text(0.02, 0.98, f'n = {n_stations} stations',
                transform=ax.transAxes, fontsize=legend_fontsize, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5, ec='none'))

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(axis='both', linestyle=':', alpha=0.4)

        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Model & Settings",
                  title_fontsize=label_fontsize-2, fontsize=legend_fontsize, labelspacing=1.2)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

        if save_path:
            fig.savefig(save_path)
        plt.close(fig)

    def plot_sites_map(self, all_lakes, groups, save_path=None):
        site_locations = []
        for group_name, lakes_in_group in groups.items():
            for lake_name, lake_obj in lakes_in_group.items():
                lats = [s.latitude for s in lake_obj.stations if s.latitude is not None]
                lons = [s.longitude for s in lake_obj.stations if s.longitude is not None]
                if lats and lons:
                    site_locations.append({
                        'Site': lake_name,
                        'Group': group_name,
                        'lat': np.mean(lats),
                        'lon': np.mean(lons)
                    })

        if not site_locations:
            print("Warning: No valid site locations to plot.")
            return

        locations_df = pd.DataFrame(site_locations)
        gdf = gpd.GeoDataFrame(
            locations_df,
            geometry=gpd.points_from_xy(locations_df.lon, locations_df.lat),
            crs="EPSG:4326"
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        world.plot(ax=ax, color='#e0e0e0', edgecolor='white')

        gdf.plot(
            ax=ax,
            column='Group',
            categorical=True,
            legend=True,
            markersize=250,
            edgecolor='black',
            linewidth=0.7,
            zorder=10
        )

        ax.set_xlim(-20, 40)
        ax.set_ylim(35, 75)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title("Overview of Study Regions", fontsize=20, pad=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.get_legend().set_title('Region')

        plt.show()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved site map to {save_path}")
        plt.close(fig)

    def plot_group_summary_bars(self, all_lakes, groups, key_variables=None, save_path_prefix=None):
        if key_variables is None:
            key_variables = ['zeu', 'chl_a_surface', 'pp_daily_depth_integrated']

        group_stats = {}
        for group_name, lakes_in_group in groups.items():
            avg_stats = {}
            for var in key_variables:
                values = [s.variables[var]['value'] for l in lakes_in_group.values() for s in l.stations if var in s.variables and s.variables[var]['value'] is not None and np.isfinite(s.variables[var]['value'])]
                avg_stats[var] = np.mean(values) if values else 0
            group_stats[group_name] = avg_stats

        stats_df = pd.DataFrame(group_stats)
        global_maxes = stats_df.max(axis=1)

        color_map = {
            'zeu': 'darkorange',
            'chl_a_surface': 'green',
            'pp_daily_depth_integrated': 'black'
        }
        plot_order = [v for v in key_variables if v in color_map]

        for group_name in groups.keys():
            if group_name not in stats_df.columns: continue

            fig, ax = plt.subplots(figsize=(2.5, 4), dpi=300)

            for i, var in enumerate(plot_order):
                raw_value = stats_df.loc[var, group_name]
                scaled_value = 0.05 + 0.95 * (raw_value / global_maxes[var])

                ax.bar(i, scaled_value, color=color_map[var], edgecolor='black', width=0.8)
                ax.text(i, scaled_value + 0.02, f'{raw_value:.1f}', ha='center', fontsize=12, fontweight='bold')

            ax.set_title(f'{group_name}', fontsize=16, pad=20)

            ax.set_xticks(range(len(plot_order)))
            ax.set_xticklabels([v.replace('_', ' ').replace(' surface', '').title() for v in plot_order], rotation=45, ha='right', fontsize=12)
            ax.set_ylim(0, 1.2)
            ax.get_yaxis().set_visible(False)
            sns.despine(ax=ax, left=True, bottom=True)

            plt.tight_layout()
            plt.show()

            if save_path_prefix:
                save_path = f"{save_path_prefix}_{group_name}.svg"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight', transparent=True)
                print(f"Saved infographic bar chart to {save_path}")
            plt.close(fig)

    def plot_single_model_timeseries(self, lake_object, model_name_to_plot, save_path=None):
        print(f"Plotting {lake_object.name} using model '{model_name_to_plot}'")

        records = []
        for station in sorted(lake_object.stations, key=lambda s: s.date):
            measured_val = station.variables.get('pp_daily_depth_integrated', {}).get('value')
            target_model = next((m for m in station.models if m.name == model_name_to_plot), None)
            model_val = target_model.model_pp if target_model and hasattr(target_model, 'model_pp') else np.nan
            records.append({
                'date': station.date,
                'measured': measured_val,
                'modeled': model_val
            })

        if not records:
            print(f"No station data found for {lake_object.name}.")
            return

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        if df['modeled'].isnull().all():
            print(f"Error: No model data found for name '{model_name_to_plot}' in {lake_object.name}.")
            return

        df.dropna(subset=['measured'], inplace=True)
        if df.empty:
            print(f"No valid measurement data to plot for {lake_object.name}.")
            return

        smoothed_measured = df['measured'].rolling(window='120D', min_periods=2, center=True).mean()
        smoothed_model = df['modeled'].rolling(window='120D', min_periods=2, center=True).mean()

        fig, ax = plt.subplots(figsize=(16, 8))
        model_color = '#d62728'
        measured_color = 'black'

        for date, row in df.dropna(subset=['measured', 'modeled']).iterrows():
            ax.plot([date, date], [row['measured'], row['modeled']],
                    color='grey',
                    linewidth=0.8,
                    linestyle='-',
                    alpha=0.6,
                    zorder=1)

        ax.scatter(df.index, df['measured'], s=60, facecolors='none',
                   edgecolors=measured_color, alpha=0.8, label='Measured (Raw)', zorder=3)
        ax.scatter(df.index, df['modeled'].dropna(), s=60, marker='x',
                   color=model_color, alpha=0.9, label='Model (Raw)', zorder=2)

        ax.plot(smoothed_measured.index, smoothed_measured.values, color=measured_color,
                linewidth=2.0, alpha=0.8, label='Measured (120-Day Trend)', zorder=4)
        ax.plot(smoothed_model.index, smoothed_model.values, color=model_color,
                linewidth=2.0, alpha=0.8, linestyle='-', label='Model (120-Day Trend)', zorder=4)

        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_ylabel('PP [mg C m⁻² d⁻¹]', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(f'Time Series Comparison for {lake_object.name}\nModel: {model_name_to_plot}', fontsize=16)

        handles, labels = ax.get_legend_handles_labels()
        order = [2, 0, 3, 1]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize=10)

        ax.margins(x=0.02)
        fig.autofmt_xdate()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Figure saved to: {save_path}")

        plt.show()
        plt.close(fig)

    def plot_longterm_detailed_figure(self, lake1_obj, lake2_obj, model_name, data_type, save_path=None):
        if data_type not in ['daily', 'hourly']:
            print("Error: data_type must be 'daily' or 'hourly'.")
            return

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 14), sharex=True)
        fig.suptitle(f'Time Series Comparison ({data_type.capitalize()} PP, 2000-2020)', fontsize=20, y=0.96)

        for i, lake_obj in enumerate([lake1_obj, lake2_obj]):
            ax = axes[i]

            if data_type == 'daily':
                pp_variable = 'pp_daily_depth_integrated'
                ax.set_ylabel('Daily PP [mg C m⁻² d⁻¹]')
            else:
                pp_variable = 'pp_hourly_depth_integrated'
                ax.set_ylabel('Hourly PP [mg C m⁻² h⁻¹]')

            records = []
            for station in sorted(lake_obj.stations, key=lambda s: s.date):
                measured_val = station.variables.get(pp_variable, {}).get('value')
                target_model = next((m for m in station.models if m.name == model_name), None)

                if data_type == 'daily':
                    model_val = target_model.model_pp if target_model and hasattr(target_model, 'model_pp') else np.nan
                else:
                    model_val = np.nan
                    if target_model and hasattr(target_model, 'model_pp_t'):
                        try:
                            inc_hours = station.variables.get('incubation_hours', {}).get('value')
                            if inc_hours is not None and len(inc_hours) > 0:
                                model_val = np.mean(target_model.model_pp_t[inc_hours])
                        except (IndexError, TypeError): pass

                records.append({'date': station.date, 'measured': measured_val, 'modeled': model_val})

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            df = df.loc['2000-01-01':'2020-12-31']
            df.dropna(subset=['measured'], inplace=True)

            if df.empty:
                ax.text(0.5, 0.5, 'No data in selected range', ha='center', va='center', transform=ax.transAxes)
                continue

            smoothed_measured = df['measured'].rolling(window='90D', min_periods=2, center=True).mean()
            smoothed_model = df['modeled'].rolling(window='90D', min_periods=2, center=True).mean()

            model_color = '#d62728'
            measured_color = 'black'

            for date, row in df.dropna(subset=['measured', 'modeled']).iterrows():
                ax.plot([date, date], [row['measured'], row['modeled']],
                        color='grey', linewidth=0.7, linestyle='-', alpha=0.5, zorder=1)

            ax.scatter(df.index, df['measured'], s=25, facecolors='none',
                       edgecolors=measured_color, alpha=0.7, label='Measured (Raw)', zorder=3)
            ax.scatter(df.index, df['modeled'].dropna(), s=25, marker='x',
                       color=model_color, alpha=0.7, label='Model (Raw)', zorder=2)

            ax.plot(smoothed_measured.index, smoothed_measured.values, color=measured_color,
                    linewidth=2.0, label='Measured (90-Day Trend)', zorder=4)
            ax.plot(smoothed_model.index, smoothed_model.values, color=model_color,
                    linewidth=2.0, linestyle='-', label='Model (90-Day Trend)', zorder=4)

            ax.set_title(lake_obj.name, fontsize=14)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.margins(x=0.01)

        handles, labels = axes[0].get_legend_handles_labels()
        order = [2, 0, 3, 1]
        axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Figure saved to: {save_path}")

        plt.show()
        plt.close(fig)

    def plot_arst_verification_scatter(self, all_lakes_verification, model_ref_name='Arst_Reference', model_impl_name='Arst_Implementation'):
        x_values, y_values = [], []
        for lake in all_lakes_verification.values():
            for station in lake.stations:
                model_results = {}
                for model in station.models:
                    if hasattr(model, 'pp_tz_meas') and hasattr(model, 'pp_measurement_depths'):
                        pp_2d = model.pp_tz_meas
                        depths = model.pp_measurement_depths
                        if pp_2d is not None and depths is not None:
                            pp_1d = np.array(pp_2d).mean(axis=1)
                            integrated_pp = np.trapz(pp_1d, np.array(depths))
                            model_results[model.name] = integrated_pp
                if model_ref_name in model_results and model_impl_name in model_results:
                    x_values.append(model_results[model_ref_name])
                    y_values.append(model_results[model_impl_name])

        if not x_values:
            print(f"No paired model data found for Arst verification.")
            return

        x_values, y_values = np.array(x_values), np.array(y_values)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_clean, y_clean = x_values[mask], y_values[mask]
        num_valid_points = len(x_clean)

        stats_text = "N/A"
        if num_valid_points >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            if not np.isnan(slope):
                r_squared = r_value**2
                stats_text = f'$R^2 = {r_squared:.4f}$\nSlope = {slope:.4f}'

        plt.figure(figsize=(7, 7))
        plt.scatter(x_values, y_values, alpha=0.4, edgecolor='k', s=40)
        plt.xscale('log'); plt.yscale('log')
        valid_line_pts = (x_values > 0) & (y_values > 0)
        if np.any(valid_line_pts):
            min_val = min(np.min(x_values[valid_line_pts]), np.min(y_values[valid_line_pts])) * 0.9
            max_val = max(np.max(x_values[valid_line_pts]), np.max(y_values[valid_line_pts])) * 1.1
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.title('Verification of Framework Configuration (ABB) (n=967)', fontsize=16)
        plt.xlabel('Reference Implementation (Arst et al., 2008) [mg C m$^{-2}$ h$^{-1}$]', fontsize=12)
        plt.ylabel('Modular Framework Implementation [mg C m$^{-2}$ h$^{-1}$]', fontsize=12)

        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.tight_layout()
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'figures')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Figure_arst_verification.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()

    def plot_vgpm_comparison_scatter(self, all_lakes_comparison, model_x_name, model_y_name):
        x_values, y_values = [], []
        for lake in all_lakes_comparison.values():
            for station in lake.stations:
                model_results = {}
                for model in station.models:
                    if hasattr(model, 'model_pp'):
                        integrated_pp = model.model_pp
                        model_results[model.name] = integrated_pp
                if model_x_name in model_results and model_y_name in model_results:
                    x_values.append(model_results[model_x_name])
                    y_values.append(model_results[model_y_name])

        if not x_values:
            print(f"No paired model data found for VGPM comparison.")
            return

        x_values, y_values = np.array(x_values), np.array(y_values)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_clean, y_clean = x_values[mask], y_values[mask]
        num_valid_points = len(x_clean)

        stats_text = "N/A"
        if num_valid_points >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            if not np.isnan(slope):
                r_squared = r_value**2
                stats_text = f'$R^2 = {r_squared:.4f}$\nSlope = {slope:.4f}'

        plt.figure(figsize=(7, 7))
        plt.scatter(x_values, y_values, alpha=0.4, edgecolor='k', s=40)
        plt.xscale('log'); plt.yscale('log')
        valid_line_pts = (x_values > 0) & (y_values > 0)
        if np.any(valid_line_pts):
            min_val = min(np.min(x_values[valid_line_pts]), np.min(y_values[valid_line_pts])) * 0.9
            max_val = max(np.max(x_values[valid_line_pts]), np.max(y_values[valid_line_pts])) * 1.1
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.title('Verification of Framework Configuration (ADD) (n=970)', fontsize=16)
        plt.xlabel('Standard depth-integrated VGPM [mg C m$^{-2}$ d$^{-1}$]', fontsize=12)
        plt.ylabel('Framework Implementation (ADD) [mg C m$^{-2}$ d$^{-1}$]', fontsize=12)
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.tight_layout()
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'figures')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Figure_vgpm_comparison.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
