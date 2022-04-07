import logging
from math import comb
import os

import numpy as np
import pandas as pd


# noinspection PyBroadException
class TPascal:

    def __init__(self, verbose, threshold_value=0.02):
        """
        Pascal's Triangle object responsible for preparing the data and performing calculations

        :param verbose: True if debug log should be given
        :type verbose: Boolean
        :param threshold_value: Value of the threshold
        :type threshold_value: float
        """

        self.threshold_value = threshold_value
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

        :param n: Number of carbon atoms in the molecule
        :type n: int
        :param k: Isotopologue
        :type k: int
        :param p: Abundance of labelled precursor
        :type p: float

        :return: Abundance of isotopologue k
        """
        n = n - 1
        abundance = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        return abundance

    @staticmethod
    def get_abundance_list(n, p):
        """
        Build a list of abundances for each carbon of the molecule

        :param n: number of carbons in the molecule
        :type n: int
        :param p: Abundance of labelled precursor
        :type p: float

        :return: list of abundances for each isotopologue of the molecule
        """

        abundances = [TPascal.get_abundance(n, k, p) for k in range(n)]
        return abundances

    # We can now start building the methods that will control the data manipulations
    def import_isocor_data(self, path):
        """
        Import the data from Isocor output file

        :param path: path to Isocor file
        :type path: str
        """

        try:
            # Read data and select usefull columns
            self.data = pd.read_csv(path, sep="\t")  # Isocor output is .tsv
            self.data = self.data[["sample", "metabolite", "area"]]
            # Create unique identifier for each row of the df for the upcoming merge
            self.data["ID"] = list(zip(self.data["sample"], self.data["metabolite"]))
        except Exception as err:
            self.logger.error(f"Error while reading file: {err}")
        else:
            self.logger.info("Data loaded")

    def prep_data(self):
        """Clean the data before performing calculations"""

        try:
            # Calculate total ares for each metabolite in the given sample
            total_areas = self.data.drop(["ID"], axis=1).groupby(by=["sample", "metabolite"]).sum()
            # Select indices and values to create the dictionary from which the new dataframe will be created
            indices = list(total_areas.index)
            values = list(total_areas.values)
            my_dict = {ind: val for ind, val in zip(indices, values)}
            total_areas_df = pd.DataFrame.from_dict(my_dict, orient="index", columns=["Total_Area"])
            # Create the identifiers for the upcoming merge
            total_areas_df.index.rename("ID", inplace=True)
            self.df_ready = pd.merge(self.data, total_areas_df, on="ID")
            self.sample_list = [i for i in self.df_ready["sample"].unique()]
            self.metabolite_list = [i for i in self.df_ready["metabolite"].unique()]
        except Exception:
            self.logger.exception("There was an error while cleaning the data")
        else:
            self.logger.info("Data clean and ready")

    def calculate_ratios(self):
        """Calculate Area/Total_Area ratios for each row"""

        try:
            self.df_ready["Ratio"] = self.df_ready["area"] / self.df_ready["Total_Area"]
            # Check if ratios are over the threshold value
            self.df_ready["Thresholds"] = self.df_ready.Ratio > self.threshold_value
        except Exception:
            self.logger.exception("Error while calculating ratios")
        else:
            self.logger.info("Experimental ratios have been calculated")

    # TODO: Donner option pour pouvoir plotter aires de deux manips différentes (ancien + nouveau par exemple)

    def calculate_mean_ratios(self):
        """
        Get the mean and standard deviation of each isotopologue ratio
        of a given metabolite between all samples
        """

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

    def get_isonumbs(self, precursor_abundance=0.513):
        """Calculate theoretical abundances of each isotopologue for each metabolite in each sample"""

        dataframes = []
        for sample in self.sample_list:
            for metabolite in self.metabolite_list:
                tmp_df = self.df_ready[(self.df_ready["sample"] == sample) & (
                        self.df_ready["metabolite"] == metabolite)].copy()
                abundances = TPascal.get_abundance_list(len(tmp_df["area"]), precursor_abundance)
                tmp_df["Theoretical_Ratios"] = abundances
                # Give the isotopologues the standard M + n format
                tmp_df["isotopologue"] = [f"M{i}" for i in range(len(tmp_df.metabolite))]
                dataframes.append(tmp_df)
        self.df_ready = pd.concat(dataframes)
        self.logger.info("Theoretical ratios have been calculated")

    def calculate_biases(self):
        """Calculate bias between experimental ratio and theoretical values"""

        self.df_ready["Bias (%)"] = abs((self.df_ready["Ratio"] - self.df_ready["Theoretical_Ratios"]) * 100)
        self.logger.info("Biases have been calculated")

    def calculate_mean_biases(self):
        """Calculate means and SDs for the calculated biases"""

        dataframes = []
        for sample in self.sample_list:
            for metabolite in self.metabolite_list:
                tmp_df = self.df_ready[(self.df_ready["sample"] == sample) & (
                        self.df_ready["metabolite"] == metabolite)].copy()
                total = len(tmp_df["Bias (%)"])  # Number of biases for mean calculation
                tmp_df["Mean Bias (%)"] = tmp_df["Bias (%)"].sum() / total
                tmp_df["Mean Bias SD (%)"] = np.std(tmp_df["Bias (%)"])
                dataframes.append(tmp_df)
        self.df_ready = pd.concat(dataframes)
        self.logger.info("Mean biases have been calculated")

    def export_results(self, run_name, home, mean_ratios):
        """
        Export the final results table

        :param run_name: Name of the run (used for creating the end folder)
        :type run_name: str
        :param home: Path to root
        :type home: str
        :param mean_ratios: Should means over all samples be calculated for the ratios
        :type mean_ratios: Boolean
        """

        # Create the working directory of the run and enter
        wd = home / run_name
        wd.mkdir()
        os.chdir(wd)
        self.df_ready.drop("ID", axis=1, inplace=True)  # IDs are not needed by end user
        # Reorder the columns for the end user
        if mean_ratios:
            self.df_ready = self.df_ready[["sample", "metabolite", "isotopologue", "area", "Total_Area", "Ratio",
                                           "Theoretical_Ratios", "Mean_Ratios", "Mean_Ratios_SD", "Thresholds",
                                           "Bias (%)", "Mean Bias (%)", "Mean Bias SD (%)"]]
        else:
            self.df_ready = self.df_ready[["sample", "metabolite", "isotopologue", "area", "Total_Area", "Ratio",
                                           "Theoretical_Ratios", "Thresholds", "Bias (%)", "Mean Bias (%)",
                                           "Mean Bias SD (%)"]]
        self.df_ready.to_excel("Results.xlsx", index=False)
        os.chdir(home)  # Get back to root
        self.logger.info("Results have been generated. Check parent file for Results.xlsx")
