import io
import logging
from math import ceil
from pathlib import Path
import os
from operator import itemgetter

import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from MSPT.TrianglePascal import TPascal


# noinspection PyTypeChecker
class TpN:

    def __init__(self, verbose):

        # Initialize child logger for class instances
        self.logger = logging.getLogger("TrianglePascal.notebook.TpNotebook")
        # fh = logging.FileHandler(f"{self.run_name}.log")
        handler = logging.StreamHandler()
        if verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.home = Path(os.getcwd())
        self.run_name = None
        # Initialize Pascal's Triangle object
        self.Triangle = TPascal(False)
        # Initialize all the widgets
        self.text = widgets.Text(value="",
                                 description="Run Name",
                                 disabled=False,
                                 placeholder="")
        self.errorbars = widgets.Checkbox(value=False, description="Plot Mean Ratios", disabled=False, indent=False)
        self.upload_data_btn = widgets.FileUpload(accept='',
                                                  multiple=False,
                                                  description='Upload Isocor Data')
        self.submit_btn = widgets.Button(description='Submit data',
                                         button_style='',
                                         tooltip='Click to submit selection',
                                         icon='')
        self.make_plots_btn = widgets.Button(description='Generate plots',
                                             button_style='',
                                             tooltip='Click to submit selection',
                                             icon='')
        self.metabolite_choice_btn = widgets.Button(description='Submit Metabolites',
                                                    button_style='',
                                                    tooltip='Click to submit selection',
                                                    icon='')
        self.save_plots_btn = widgets.Button(description="Save plots")
        # Initialize outputs
        self.out = widgets.Output()
        self.metselect_out = widgets.Output()
        self.plot_out = widgets.Output()
        # Initialize extras
        self.dropdown_list = []
        self.plot_dfs = []
        self.export_figures = []
        self.dropdown_options = {}
        self.in_threshold = {}
        # Initialize button click event
        self.metabolite_choice_btn.on_click(self._build_dropdowns)

    def reset(self, verbose):
        """Function to reset the object in notebook
        (only for notebook use because without the function
        cell refresh doesn't reinitialize the object)"""

        os.chdir(self.home)
        self.__init__(verbose)

    @staticmethod
    def _half_list(array):
        """
        Quick method for dividing a list into two lists

        :param array: array to split
        :return: two equivalent arrays (one longer than the other if
        """

        n = len(array)
        half = n / 2  # py3
        if isinstance(half, float):
            return array[:ceil(half)], array[n - int(half):]
        else:
            return array[:half], array[n - half:]

    @staticmethod
    def _parse_df_list(dfs):
        """
        Method for splitting the dfs in two for plotting each subplot of the pdf pages

        """

        df1, df2 = [], []
        while dfs:
            try:
                df1.append(dfs.pop(0))
                df2.append(dfs.pop(0))
            except IndexError:
                break
        return df1, df2

    @staticmethod
    def _build_axes(metabolite, real, theory, labels, width=0.2, ax=None, yerr=None):
        """
        Barplot builder

        :param metabolite:
        :param real:
        :param theory:
        :param labels:
        :param width:
        :param ax:
        :param yerr:
        :return:
        """

        x = np.arange(len(labels))
        if ax is None:
            plt.gca()
        rects1 = ax.bar(x - width / 2, real, width, label="Experimental", yerr=yerr)
        rects2 = ax.bar(x + width / 2, theory, width, label="Theory", yerr=yerr)
        ax.set_ylabel("Recorded Area")
        ax.set_title(f"{metabolite}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height - (height / 90)),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize="xx-small")

        if yerr is not None:
            autolabel(rects1)
            autolabel(rects2)
        return rects1, rects2

    @staticmethod
    def _build_figure(df1, df2=None, yerr=False):
        """
        Figure builder

        :param df1:
        :param df2:
        :param yerr:
        :return:
        """
        if df2 is None:
            if yerr:
                yerr1 = df1.Mean_Ratios_SD
            else:
                yerr1 = None
            fig, ax = plt.subplots()
            fig.set_size_inches([5, 4])
            TpN._build_axes(list(df1.metabolite)[0],
                            df1.Ratio,
                            df1.Theoretical_Ratios,
                            df1.isotopologue,
                            ax=ax, yerr=yerr1)
        else:
            if yerr:
                yerr1 = df1.Mean_Ratios_SD
                yerr2 = df2.Mean_Ratios_SD
            else:
                yerr1, yerr2 = None, None

            fig, (ax1, ax2) = plt.subplots(nrows=2)
            fig.set_size_inches([5, 4])
            TpN._build_axes(list(df1.metabolite)[0],
                            df1.Ratio,
                            df1.Theoretical_Ratios,
                            df1.isotopologue,
                            ax=ax1, yerr=yerr1)
            TpN._build_axes(list(df2.metabolite)[0],
                            df2.Ratio,
                            df2.Theoretical_Ratios,
                            df2.isotopologue,
                            ax=ax2, yerr=yerr2)
            ax1.legend(prop={'size': "x-small"})
            ax2.legend(prop={'size': "x-small"})
            fig.tight_layout()
        return fig

    def initialize_widgets(self):
        """Display initial widgets"""

        display(self.text,
                self.errorbars,
                self.upload_data_btn,
                self.submit_btn)

    def _get_data(self, button):
        """
        Get data from button and return it

        :param button:
        :return:
        """

        try:
            data = next(iter(button.value))
        except StopIteration:
            return f"No file loaded in {button}"
        data_content = button.value[data]['content']
        with open('../myfile', 'wb') as f:
            f.write(data_content)
        try:
            real_data = pd.read_csv(io.BytesIO(data_content), sep="\t")
        except Exception as e:
            self.logger.error("There was a problem reading file")
            self.logger.error(e)
        else:
            return real_data

    # noinspection PyUnresolvedReferences
    def _submit_data(self, event):
        """Submit data event for submit button click"""

        self.run_name = self.text.value
        self.Triangle.data = self._get_data(self.upload_data_btn)
        self.Triangle.data = self.Triangle.data[["sample", "metabolite", "area"]]
        self.Triangle.data["ID"] = list(zip(self.Triangle.data["sample"], self.Triangle.data["metabolite"]))
        self.Triangle.prep_data()
        self.Triangle.calculate_ratios()
        self.Triangle.get_isonumbs()
        if len(self.Triangle.sample_list) > 1:
            self.Triangle.calculate_mean_ratios()
            mean_ratios = True
        else:
            mean_ratios = False
        self.Triangle.calculate_biases()
        self.Triangle.calculate_mean_biases()
        self.Triangle.export_results(self.run_name, self.home, mean_ratios)
        self.metabolite_choice = widgets.SelectMultiple(options=self.Triangle.metabolite_list,
                                                        description="Metabolites",
                                                        value=[self.Triangle.metabolite_list[0]],
                                                        disabled=False)
        with self.metselect_out:
            display(self.metabolite_choice,
                    self.metabolite_choice_btn)
        self.Triangle.logger.info("Done processing data")

    def _build_dropdowns(self, event):
        """Function to create dropdowns per metabolite for selection of isotopologue to plot"""

        # Prepare to clear output if receives new event from metabolite choice button
        self.out.clear_output()
        # Re-initialize so that when button is pressed once more the dropdowns are refreshed with new widget count
        self.dropdown_options = {}
        self.in_threshold = {}
        self.dropdown_list = []
        tmp_df = self.Triangle.df_ready[self.Triangle.df_ready["sample"] == self.Triangle.sample_list[0]]
        for metabolite in self.metabolite_choice.value:
            tmp_df_2 = tmp_df[tmp_df["metabolite"] == metabolite]
            # Not 1 to len +1 because isotopologues start from 0
            options = ["M" + str(n) for n in range(0, len(tmp_df_2.metabolite))]
            thresholds = [i for i in tmp_df_2.Thresholds]  # Boolean array where True is over 0.02
            self.dropdown_options.update({metabolite: options})
            self.in_threshold.update({metabolite: thresholds})
        # Generate selection widgets with metabolite names and isotopologues from dictionnary. We want to select ratio
        # values that are at least over 2% (ratio 0.02).
        for (key_opt, val_opt), (_, val_thr) in zip(self.dropdown_options.items(), self.in_threshold.items()):
            indices = [i for i, x in enumerate(val_thr) if x]  # Get isotopologues where we are over threshold
            try:
                self.logger.debug(f"value options = {val_opt}")
                self.logger.debug(f"key options = {key_opt}")
                self.dropdown_list.append(widgets.SelectMultiple(options=val_opt,
                                                                 value=list(itemgetter(*indices)(val_opt)),
                                                                 description=key_opt, disabled=False))
            # If there is no experimental data, there are no options so an error is raised. In this case we log
            # the metabolite for which it happened
            except TypeError:
                with self.plot_out:
                    self.logger.warning(f"Experimental data missing for {key_opt}")
                continue
            except Exception:
                with self.plot_out:
                    self.logger.exception(f"Error while calculating for {key_opt}")
                continue
        # Split in half for more clarity in the notebook (two columns of selection widgets)
        dropdowns1, dropdowns2 = TpN._half_list(self.dropdown_list)
        # Layout the widgets
        v_box1 = widgets.VBox(children=dropdowns1)
        v_box2 = widgets.VBox(children=dropdowns2)
        self.dropdowns = widgets.HBox(children=[v_box1, v_box2])
        with self.out:
            display(self.dropdowns,
                    self.make_plots_btn)

    def load_events(self):
        """Prepare button click events"""

        self.submit_btn.on_click(self._submit_data)
        self.make_plots_btn.on_click(self._make_ind_plots)
        self.save_plots_btn.on_click(self._save_plots)

    def _make_ind_plots(self, event):
        """Event after clicking make plots to build the individual plots"""

        # We loop on widgets for simplicity. Filtering is done using metadata from each widget
        self.plot_dfs.clear()
        self.plot_out.clear_output()
        for widget in self.dropdown_list:
            tmp_df = self.Triangle.df_ready[(self.Triangle.df_ready["sample"] == self.Triangle.sample_list[0]) &
                                            (self.Triangle.df_ready["metabolite"] == widget.description) &
                                            (self.Triangle.df_ready["isotopologue"].isin(list(widget.value)))]
            self.plot_dfs.append(tmp_df)
            if self.errorbars.value:
                yerr = tmp_df.Mean_Ratios_SD
            else:
                yerr = None
            with self.plot_out:
                fig, ax = plt.subplots()
                TpN._build_axes(widget.description,
                                tmp_df.Ratio,
                                tmp_df.Theoretical_Ratios,
                                tmp_df.isotopologue,
                                ax=ax, yerr=yerr)
                plt.show()
        with self.plot_out:
            display(self.save_plots_btn)

    def _save_plots(self, event):

        wd = self.home / self.run_name
        if not os.path.exists(wd):
            wd.mkdir()
        os.chdir(wd)
        dfs1, dfs2 = TpN._parse_df_list(self.plot_dfs)
        figures = []
        for df1, df2 in zip(dfs1, dfs2):
            fig = TpN._build_figure(df1, df2, yerr=self.errorbars.value)
            figures.append(fig)

        def last_df(dfs_1, dfs_2):
            if len(dfs_1) > len(dfs_2):
                return dfs_1[-1]
            elif len(dfs_1) < len(dfs_2):
                return dfs_2[-1]

        if len(dfs1) != len(dfs2):
            last_df = last_df(dfs1, dfs2)
            fig = TpN._build_figure(last_df, yerr=self.errorbars.value)
            figures.append(fig)
        with PdfPages("Plots.pdf") as pdf:
            for fig in figures:
                pdf.savefig(fig)
        os.chdir(self.home)
