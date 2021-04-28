import logging
from math import comb
import os

import numpy as np
import pandas as pd


class TPascal:

    def __init__(self, verbose):
        """
        Pascal's Triangle object responsible for preparing the data and performing calculations

        :param verbose:
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        if verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)

        if not self.logger.hasHandlers:
            self.logger.addHandler(handler)

        self.data = None
        self.total_areas = None
        self.df_ready = None
        self.sample_list = []
        self.metabolite_list = []
        self.thresholds = None

        self.sample_mean = True
        self.metabolite_mean = True

    @staticmethod
    def get_abundance(n, k, p):
        """
        Get the abundance of an isotopologue (k) in the isotopologic space of the molecule containing n carbons
        from the abundance (p) of labelled precursor

        :param n:
        :param k:
        :param p:
        :return:
        """
        n = n - 1
        abundance = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

        return abundance

    @staticmethod
    def get_abundance_list(n, p):
        """
        Build a list of abundances for each carbon of the molecule

        :param n:
        :param p:
        :return:
        """

        abundances = [TPascal.get_abundance(n, k, p) for k in range(n)]

        return abundances

    # We can now start building the methods that will control the data manipulations
    def import_isocor_data(self, path):
        """
        Import the data from Isocor output file

        :param path:
        :return:
        """

        try:
            self.data = pd.read_csv(path, sep="\t")  # Isocor output is .tsv
            self.data = self.data[["sample", "metabolite", "area"]]
            self.data["ID"] = list(zip(self.data["sample"], self.data["metabolite"]))
        except Exception as err:
            self.logger.error(f"Error while reading file: {err}")
        else:
            self.logger.info("Data loaded")

    def prep_data(self):

        try:
            total_areas = self.data.drop(["ID"], axis=1).groupby(by=["sample", "metabolite"]).sum()
            indices = list(total_areas.index)
            values = list(total_areas.values)
            my_dict = {ind: val for ind, val in zip(indices, values)}
            total_areas_df = pd.DataFrame.from_dict(my_dict, orient="index", columns=["Total_Area"])
            total_areas_df.index.rename("ID", inplace=True)
            self.df_ready = pd.merge(self.data, total_areas_df, on="ID")

            self.sample_list = [i for i in self.df_ready["sample"].unique()]
            self.metabolite_list = [i for i in self.df_ready["metabolite"].unique()]

        except Exception as e:
            self.logger.error(f"There was an error while cleaning the data: {e}")
        else:
            self.logger.info("Data clean and ready")

    def calculate_ratios(self):

        try:
            self.df_ready["Ratio"] = self.df_ready["area"] / self.df_ready["Total_Area"]
            self.df_ready["Thresholds"] = self.df_ready.Ratio > 0.02
        except Exception as e:
            self.logger.error(f"Error during division. Error: {e}")
        else:
            self.logger.info("Experimental ratios have been calculated")

    def calculate_mean_ratios(self):

        dataframes = []

        for metabolite in self.metabolite_list:
            for isotop in self.df_ready[self.df_ready["metabolite"] == metabolite]["isotopologue"].unique():
                tmp_df = self.df_ready[(self.df_ready["metabolite"] == metabolite) & (
                        self.df_ready["isotopologue"] == isotop)].copy()

                self.logger.debug(f"tmp_df:\n{tmp_df}")

                tmp_df["Mean_Ratios"] = tmp_df["Ratio"].sum() / len(tmp_df["Ratio"])
                tmp_df["Mean_Ratios_SD"] = np.std(tmp_df["Ratio"])
                dataframes.append(tmp_df)

        self.df_ready = pd.concat(dataframes)

    def get_isonumbs(self):

        dataframes = []
        for sample in self.sample_list:
            for metabolite in self.metabolite_list:

                tmp_df = self.df_ready[(self.df_ready["sample"] == sample) & (
                        self.df_ready["metabolite"] == metabolite)].copy()
                abundances = TPascal.get_abundance_list(len(tmp_df["area"]), 0.5)
                tmp_df["Theoretical_Ratios"] = abundances
                tmp_df["isotopologue"] = [f"M{i}" for i in range(len(tmp_df.metabolite))]
                dataframes.append(tmp_df)

        self.df_ready = pd.concat(dataframes)

        self.logger.info("Theoretical ratios are calculated")

    def calculate_biases(self):

        self.df_ready["Bias (%)"] = abs((self.df_ready["Ratio"] - self.df_ready["Theoretical_Ratios"]) * 100)

        self.logger.info("Biases have been calculated")

    def calculate_mean_biases(self):

        dataframes = []

        for sample in self.sample_list:
            for metabolite in self.metabolite_list:
                tmp_df = self.df_ready[(self.df_ready["sample"] == sample) & (
                        self.df_ready["metabolite"] == metabolite)].copy()
                total = len(tmp_df["Bias (%)"])
                tmp_df["Mean Bias (%)"] = tmp_df["Bias (%)"].sum() / total
                tmp_df["Mean Bias SD (%)"] = np.std(tmp_df["Bias (%)"])
                dataframes.append(tmp_df)

        self.df_ready = pd.concat(dataframes)

        self.logger.info("Mean biases have been calculated")

    def export_results(self, run_name, home):
        """
        Export the final results table

        :param run_name:
        :param home:
        :return:
        """

        wd = home / run_name
        wd.mkdir()
        os.chdir(wd)

        self.df_ready.drop("ID", axis=1, inplace=True)
        self.df_ready = self.df_ready[["sample", "metabolite", "isotopologue", "area", "Total_Area", "Ratio",
                                       "Theoretical_Ratios", "Mean_Ratios", "Mean_Ratios_SD","Thresholds", "Bias (%)",
                                       "Mean Bias (%)", "Mean Bias SD (%)"]]
        self.df_ready.to_excel("Results.xlsx", index=False)

        os.chdir(home)

        self.logger.info("Results have been generated. Check parent file for Results.xlsx")

