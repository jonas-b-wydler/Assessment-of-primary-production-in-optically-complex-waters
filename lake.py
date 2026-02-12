import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from utility import save_plot
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


class Lake:
    def __init__(self, name, stations=None):
        self.name = name
        self.stations = stations if stations is not None else []
        self.filtered_stations = []

    def add_station(self, station):
        self.stations.append(station)


class depth_resolved_lake(Lake):
    def __init__(self, name):
        super().__init__(name)

    def filter_stations(self):
        required_vars = [
            'pp_hourly_depth_integrated',
            'chl_a_surface',
            'chl_a_depth_resolved',
            'secchi',
            'par',
            'par_max',
            'surface_pb_opt',
            'zeu',
            'incubation_hours'
        ]

        filtered_stations = []

        for station in self.stations:
            if all(
                var in station.variables and
                station.variables[var]['value'] is not None and
                (
                    (isinstance(station.variables[var]['value'], pd.Series) and not station.variables[var]['value'].isna().any()) or
                    (isinstance(station.variables[var]['value'], pd.DataFrame) and not station.variables[var]['value'].isna().any().any()) or
                    (isinstance(station.variables[var]['value'], np.ndarray) and not np.isnan(station.variables[var]['value']).any()) or
                    (isinstance(station.variables[var]['value'], (float, int, np.float64)) and not pd.isna(station.variables[var]['value']))
                )
                for var in required_vars
            ):
                secchi_depth = station.variables['secchi']['value']
                if isinstance(secchi_depth, (float, int, np.float64)) and secchi_depth > 0.05:
                    filtered_stations.append(station)

        self.stations = filtered_stations
        self.filtered_stations = filtered_stations

    def plot_modeled_curve(self, variable_name="pp_hourly_depth_resolved"):
        """
        Plot observed vs. modeled depth-resolved primary production for each station
        and model in this lake, using a proper secondary axis for Chlorophyll-a.
        Uses model_pp_z / dl (not pp_tz_meas), so works for all depth-resolved models.
        """
        for st in self.stations:
            if variable_name not in st.variables:
                continue

            # Observed PP profile
            obs_df = st.variables[variable_name]["value"]
            obs_z = obs_df.index.to_numpy(float)
            obs_pp = obs_df.to_numpy(float).flatten()
            obs_int = np.trapz(obs_pp, obs_z)

            # Date string for file naming and titles
            date_str = st.date.strftime("%Y%m%d") if getattr(st, "date", None) else "nodate"

            for mdl in st.models:
                if not getattr(mdl, "is_depth_resolved", False):
                    continue

                # Get Modeled PP profile and divide by DL for correct units
                dl_val = st.variables["dl"]["value"]
                mdl_z = mdl.depths
                mdl_pp = mdl.model_pp_z / dl_val
                mdl_int = np.trapz(mdl_pp, mdl_z)

                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

                color_obs_pp = 'C0'
                color_mod_pp = 'C1'

                p1, = ax.plot(obs_pp, obs_z, "o-", label="Observed PP", color=color_obs_pp)
                p2, = ax.plot(mdl_pp, mdl_z, "x--", label=f"Modeled PP ({mdl.name})", color=color_mod_pp)

                ax.set_xlabel("PP (mg C m⁻³ h⁻¹)", color=color_obs_pp)
                ax.tick_params(axis='x', labelcolor=color_obs_pp)
                ax.set_ylabel("Depth (m)")
                ax.invert_yaxis()
                ax.grid(True)

                handles = [p1, p2]

                if "chl_a_depth_resolved" in st.variables:
                    ax2 = ax.twiny()
                    color_chl = 'C2'
                    chl_df = st.variables["chl_a_depth_resolved"]["value"]
                    p3, = ax2.plot(chl_df.to_numpy(float), chl_df.index.to_numpy(float),
                                   "s-.", label="Observed Chl-a", color=color_chl)
                    ax2.set_xlabel("Chl-a (mg m⁻³)", color=color_chl)
                    ax2.tick_params(axis='x', labelcolor=color_chl)
                    handles.append(p3)

                ax.legend(handles=handles, loc="lower right")
                ax.set_title(f"{st.station_id} | {date_str} | {self.name} | {mdl.name}")

                txt = (
                    f"Integrated PP:\n"
                    f"  Obs: {obs_int:.2f}\n"
                    f"  Mod: {mdl_int:.2f}\n\n"
                    f"Error Metrics:\n"
                    f"  MAPE: {mdl.mape:.1f}%\n"
                    f"  Corr: {mdl.pattern_correlation:.2f}\n\n"
                    f"Case: {mdl.case}"
                )
                ax.text(
                    0.95, 0.25, txt, transform=ax.transAxes,
                    va="bottom", ha="right",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5")
                )

                plt.tight_layout()
                plt.show()

                model_subfolder = f"depth_profiles/{mdl.name.replace(' ', '_')}"
                plot_filename = f"{st.station_id}_{date_str}_{mdl.name.replace(' ', '_')}"
                save_plot(fig, lake_name=self.name, title=plot_filename, subfolder=model_subfolder)
                plt.close(fig)

    

    def reset_outliers(self):
        for station in self.stations:
            station.is_outlier = False
            station.outlier_distance = None
            station.outlier_issues = []

    def derivative_pca_outlier_check(
        self,
        variable_pp='pp_hourly_depth_resolved',
        variable_chla='chl_a_depth_resolved',
        variable_zeu='zeu',
        distance_cut=None,
        distance_q=0.97,
        window_r=2,
        r_thresh=0.4,
        remove_amplitude=True,
        do_plot=True,
        skip_preflagged=True,
        exclude=1
    ):
        station_refs, features = [], []
        pp_cache, ch_cache, depth_cache = [], [], []
        ref_len = None

        for st in self.stations:
            if skip_preflagged and getattr(st, 'is_outlier', False):
                continue

            pp_s = st.variables.get(variable_pp, {}).get('value')
            ch_s = st.variables.get(variable_chla, {}).get('value')
            if pp_s is None or ch_s is None \
               or not np.array_equal(pp_s.index, ch_s.index):
                continue

            depth = pp_s.index.to_numpy(float)
            pp = pp_s.to_numpy(float)
            ch = ch_s.to_numpy(float)

            if ref_len is None:
                ref_len = len(pp)
            if len(pp) != ref_len:
                continue

            pp_norm = pp / pp.max() if remove_amplitude and pp.max() > 0 else pp
            spl = UnivariateSpline(depth, pp_norm, s=0)
            d1 = spl.derivative(1)(depth)[exclude:-exclude]
            d2 = spl.derivative(2)(depth)[exclude:-exclude]
            features.append(np.r_[d1, d2])

            station_refs.append(st)
            pp_cache.append(pp)
            ch_cache.append(ch)
            depth_cache.append(depth)
            st.outlier_issues = []

        if len(features) < 2:
            print("<2 valid profiles – PCA skipped>")
            return [], None

        print("Rule-based flags so far:",
              sum(getattr(st, "is_outlier", False) for st in self.stations),
              "out of", len(self.stations))

        X = StandardScaler().fit_transform(np.vstack(features))
        pca = PCA(n_components=0.95, random_state=42).fit(X)
        scores = pca.transform(X)
        dist = np.linalg.norm(scores, axis=1)

        distance_cut = distance_cut if distance_cut is not None \
                       else np.quantile(dist, distance_q)
        print(f"PCA distance cut = {distance_cut:.2f}")

        print("–– PCA diagnostics ––")
        print("Profiles entering PCA:", len(dist))
        print("Distance percentiles 90–99.9:",
              np.percentile(dist, [90, 95, 97.5, 99, 99.5, 99.9]))

        outliers = []

        for depth_arr, pp_arr, ch_arr, st, dv in zip(
                depth_cache, pp_cache, ch_cache, station_refs, dist):

            is_out = dv > distance_cut
            rescued = False

            dev_pp = pp_arr - np.median(pp_arr)
            core_idx = np.arange(exclude, len(pp_arr) - exclude)
            zeu = st.variables.get(variable_zeu, {}).get('value')

            if zeu is not None:
                mask = depth_arr[core_idx] <= zeu
                idx_anom = (core_idx[mask]
                            [np.argmax(np.abs(dev_pp[core_idx[mask]]))]) \
                            if mask.any() else None
            else:
                idx_anom = None

            st.anomaly_depth = float(depth_arr[idx_anom]) if idx_anom is not None else None

            st.window_r = None
            if is_out and idx_anom is not None:
                lo = max(exclude, idx_anom - window_r)
                hi = min(len(pp_arr) - exclude, idx_anom + window_r + 1)

                pp_slice = pp_arr[lo:hi].ravel()
                ch_slice = ch_arr[lo:hi].ravel()

                if pp_slice.size > 1:
                    r, _ = pearsonr(pp_slice, ch_slice)
                    st.window_r = float(r)
                    if r >= r_thresh:
                        rescued = True
                        st.outlier_issues.append(
                            f"Rescued (window r = {r:.2f} ≥ {r_thresh})")
                    else:
                        st.outlier_issues.append(
                            f"No rescue (window r = {r:.2f} < {r_thresh})")
            elif is_out:
                reason = "Zeu missing" if zeu is None else "no spike above Zeu"
                st.outlier_issues.append(f"No rescue ({reason})")

            if rescued:
                is_out = False

            st.is_outlier = is_out
            st.outlier_distance = float(dv)
            if is_out:
                st.outlier_issues.append(
                    f"PCA_dist={dv:.2f} > {distance_cut:.2f}")
                outliers.append((st.station_id, dv))

        print(f"Flagged {len(outliers)}/{len(features)} profiles "
              f"(cut = {distance_cut:.2f})")

        if do_plot and scores.shape[1] >= 2:
            pc1, pc2 = scores[:, 0], scores[:, 1]
            plt.figure(figsize=(7, 6))
            plt.scatter(pc1, pc2, c=dist > distance_cut,
                        cmap='coolwarm', alpha=0.7)
            plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(True)
            plt.title('Derivative-PCA outlier detection')
            for x, y, dval, st in zip(pc1, pc2, dist, station_refs):
                if dval > distance_cut:
                    plt.text(x, y, st.station_id, color='red', fontsize=8)
            plt.show()

        return outliers, pca

    def check_low_chl(self, variable_pp='pp_hourly_depth_resolved', variable_chla='chla_depth_resolved'):
        for station in self.stations:
            if variable_chla not in station.variables or variable_pp not in station.variables:
                continue

            chl_arr = station.variables[variable_chla]['value']
            pp_arr = station.variables[variable_pp]['value']
            if hasattr(chl_arr, 'values'):
                chl_arr = chl_arr.values
            if hasattr(pp_arr, 'values'):
                pp_arr = pp_arr.values

            if len(chl_arr) < 3 or len(pp_arr) < 3:
                continue

            mean_pp = np.mean(pp_arr)
            threshold_pp = max(0.1 * mean_pp, 0.2)
            for i in range(1, len(pp_arr) - 1):
                if (chl_arr[i] < 0.05) and (pp_arr[i] > threshold_pp):
                    station.is_outlier = True
                    station.outlier_issues.append("Chl <0.05 with PP > threshold")
                    break

    def check_max_pp_depth(self, variable_pp='pp_hourly_depth_resolved', variable_kd='kd_par'):
        for station in self.stations:
            if variable_kd not in station.variables or variable_pp not in station.variables:
                continue

            kd_par = station.variables[variable_kd]['value']
            pp_series = station.variables[variable_pp]['value']
            if hasattr(pp_series, 'values'):
                pp_arr = pp_series.values
            else:
                pp_arr = pp_series
            if len(pp_arr) < 2:
                continue

            if hasattr(pp_series, 'index'):
                depth_arr = pp_series.index.values
            else:
                depth_arr = station.variables[variable_pp].get('depth', None)
                if depth_arr is None:
                    continue
                if hasattr(depth_arr, 'values'):
                    depth_arr = depth_arr.values

            obs_depths = np.array(depth_arr).astype(float)

            try:
                zeu = -np.log(0.01) / float(kd_par)
            except Exception as e:
                print(f"Error computing zeu for station {station.station_id}: {e}")
                continue

            max_idx = np.argmax(pp_arr)
            max_depth = obs_depths[max_idx]

            if max_depth > 1.2 * zeu:
                station.is_outlier = True
                station.outlier_issues.append("Max PP occurs at a depth deeper than 1.2×zeu")

    def plot_all_profiles_with_chla_and_zeu(
        self,
        variable_pp='pp_hourly_depth_resolved',
        variable_chla='chl_a_depth_resolved',
        variable_kd='kd_par',
        exclude=1,
        save_dir=None,
        fmt='svg'
    ):
        if not self.stations:
            print(f"{self.name}: no stations available.")
            return

        for st in self.stations:
            pp_s = st.variables[variable_pp]['value']
            ch_s = st.variables[variable_chla]['value']
            depth = pp_s.index.to_numpy(float)

            pp = pp_s.to_numpy(float)
            ch = ch_s.to_numpy(float)

            kd = float(st.variables[variable_kd]['value'])
            zeu = -np.log(0.01) / kd

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(pp, depth, 'o-', label='PP')
            ax.plot(ch, depth, 's-', label='Chl-a')
            ax.axhline(zeu, ls='--', label=f"Z_eu={zeu:.2f} m")

            anom_depth = getattr(st, 'anomaly_depth', None)
            if anom_depth is not None:
                ax.axhline(anom_depth, color='red', ls=':',
                           label=f"Anomaly at {anom_depth:.2f} m")

            ax.invert_yaxis()
            ax.set_xlabel("Value"); ax.set_ylabel("Depth (m)")
            ax.grid(True); ax.legend()

            name_str = f"{self.name} – {st.station_id}" if getattr(self, 'name', None) != st.station_id else st.station_id
            issues = "; ".join(st.outlier_issues) if st.outlier_issues else "No issues"
            ratio_lab = (f" | ratio={st.chl_pp_ratio:.2f}"
                         if getattr(st, 'chl_pp_ratio', None) is not None else "")
            date_lab = f" | {st.date:%Y%m%d}" if getattr(st, 'date', None) else ""

            fig.suptitle(f"{name_str} | {issues}{ratio_lab}{date_lab}", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save_dir:
                safe_date = st.date.strftime('%Y-%m-%d') if getattr(st, 'date', None) else "no_date"
                clean_lake = str(self.name).replace(" ", "_").replace("(", "").replace(")", "")
                clean_station = str(st.station_id).replace(" ", "_")

                filename = f"{clean_lake}_{clean_station}_{safe_date}.{fmt}"
                save_path = os.path.join(save_dir, filename)

                plt.savefig(save_path, format=fmt)
                plt.close(fig)
                print(f"Saved: {save_path}")
            else:
                plt.show()
