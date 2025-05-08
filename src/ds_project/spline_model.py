

import pymc as pm 
import pandas as pd
from patsy import dmatrix, build_design_matrices
import numpy as np
import matplotlib.pyplot as plt
import pytensor as pt
import xarray as xr

from .utils import generate_test_data


class SplineModel:
    
    def __init__(self, 
                data,
                xcols,
                ycol,
                n_knots=5):
        """
        Initialize the spline model with data and configuration.
        """

        pt.config.mode = "NUMBA"
        pt.config.cxx = ""
        for xcol in xcols:
            if xcol not in data.columns:
                raise ValueError(f"Column {xcol} not found in data.")
        if ycol not in data.columns:
            raise ValueError(f"Column {ycol} not found in data.")

        self.data = data
        self.config = {
            "x": xcols,
            "y": ycol,
            "n_knots": n_knots,
        }
        

    def _create_knot_list(self):
        """ Create a list of knots for the spline model"""

        if "n_knots" not in self.config:
            raise ValueError("n_knots not found in config")

        for xcol in self.config["x"]:

            min_point = self.data[xcol].min()
            max_point = self.data[xcol].max()

            # add 20% to max point for prediction
            max_point += 0.2 * (max_point - min_point)

            knot_list = np.linspace(min_point, max_point, self.config["n_knots"])
            self.config[f"{xcol}_knot_list"] = knot_list
            self.config[f"{xcol}_knot_max"] = max_point
            self.config[f"{xcol}_knot_min"] = min_point
            

    def _build_design_matrix(self, 
                            number_of_years_predict=200):
        """
        Build a design matrix for the spline model.
        This will have shape (n_samples, n_knots + 2) because we're using a cubic spline,
        are ignoring the first and last knots, and have an intercept term.

        Need to have created knot_list first
        """
        for xcol in self.config["x"]:
            if f"{xcol}_knot_list" not in self.config:
                raise ValueError(f"{xcol}_knot_list not found in config")
        

        matrix_dict = {}

        model_string = []

        self.config["matrix_indices"] = {}
        index = 0
        for xcol in self.config["x"]:
            model_string.append(f"bs({xcol}, knots={xcol}_knot_list, degree=3, lower_bound={self.config[f"{xcol}_knot_min"]}, upper_bound={self.config[f"{xcol}_knot_max"]},include_intercept=True)")
            matrix_dict[f"{xcol}_knot_list"] = self.config[f"{xcol}_knot_list"][1:-1]
            matrix_dict[f"{xcol}"] = self.data[xcol]

            self.config["matrix_indices"][xcol] = np.arange(index*(self.config["n_knots"] + 2), (index+1)*(self.config["n_knots"] + 2))
            index += 1
        model_string = " + ".join(model_string)
        model_string = model_string + " - 1"
        X = dmatrix(
            model_string,
            matrix_dict
        )

        self.config["design_matrix"] = X



    def _build_prediction_set(self, predict=False):
        new_data = {}
        for xcol in self.config["x"]: 
            max = self.data[xcol].max()
            print(max)
            if predict:
                max = self.data[xcol].max() + 0.2 * (self.data[xcol].max() - self.data[xcol].min())       
            
            new_data[xcol] = np.linspace(
                self.data[xcol].min(),
                max,
                1000
            )

        return(new_data)

    def visualise_design_matrix(self,
                                x):
        """Visualise the design matrix"""

        self._create_knot_list()
        self._build_design_matrix()

        if "design_matrix" not in self.config:
            raise ValueError("design_matrix not found in config")
        
        if x not in self.config["x"]:
            raise ValueError(f"x {x} not found in config")
        
        prediction_set = self._build_prediction_set(predict=True)

        for xcol in self.config["x"]:
            prediction_set[f"{xcol}_knot_list"] = self.config[f"{xcol}_knot_list"][1:-1]
        

        X_pred_design_matrix = build_design_matrices(
            [self.config["design_matrix"].design_info], prediction_set
        )[0]


        spline_df = (
            pd.DataFrame(X_pred_design_matrix[:, self.config["matrix_indices"][x]])
            .assign(x=prediction_set[x])
            .melt("x", var_name="spline_i", value_name="value")
        )


        colors = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

        plt.figure(figsize=(12, 8))
        for i in range(len(spline_df.spline_i.unique())):
            group_data = spline_df[spline_df["spline_i"] == i]
            plt.plot(
                group_data["x"],
                group_data["value"],
                label=f"Spline {i}",
                color=colors[i],
            )

    def conditional_effects(self, x, predict=False):

        if x not in self.config["x"]:
            raise ValueError(f"x {x} not found in config")
        
        prediction_set = self._build_prediction_set(predict)
        
        # Make sure that all other columns are set to their mean
        # for the conditional effect
        not_x_cols = [col for col in self.config["x"] if col != x]
        for not_x_col in not_x_cols:
            mean_val = np.mean(self.data[not_x_col])
            n_times = prediction_set[not_x_col].shape[0]
            prediction_set[not_x_col] = np.repeat(mean_val, n_times)
            
        
        # Use previous knot list
        for xcol in self.config["x"]:
            prediction_set[f"{xcol}_knot_list"] = self.config[f"{xcol}_knot_list"][1:-1]

        
        X_pred_design_matrix = build_design_matrices(
            [self.config["design_matrix"].design_info], prediction_set
        )[0]

        model_coords = {
            "N": np.arange(prediction_set[self.config["x"][0]].shape[0]),
            "N_knot": np.arange(self.config["n_knots"] + 2)
        }

        with self.model:

            new_data = {}
            for xcol in self.config["x"]:
                new_data[f"{xcol}_basis"] = X_pred_design_matrix[
                    :, self.config["matrix_indices"][xcol]
                ].astype("float64")
                
            
            pm.set_data(new_data, coords=model_coords)

            out_of_sample_predictions = pm.sample_posterior_predictive(
                self.idata, predictions=True, var_names=["mu"]
            ).predictions

            out_of_sample_predictions[x] = xr.DataArray(
                prediction_set[x], dims=("N")
            )

        X = out_of_sample_predictions.isel(chain=1, draw=1)[x].values


        mean = out_of_sample_predictions["mu"].mean(dim=["chain", "draw"])
        lower_95 = out_of_sample_predictions["mu"].quantile(0.025, dim=["chain", "draw"]).values
        upper_95 = out_of_sample_predictions["mu"].quantile(0.975, dim=["chain", "draw"]).values


        # Plot the summary statistics as a ribbon

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["year"], df["doy"], label="Observed Data", alpha=0.5)

        ax.plot(X, mean, label="Mean", color="black")
        ax.fill_between(X, lower_95, upper_95, color="gray", alpha=0.5, label="95% CI")


        return fig, ax
        




    def fit(self):
        """
        Fit the spline model
        """

        self._create_knot_list()
        self._build_design_matrix()

        model_coords = {
            "N": np.arange(self.data.shape[0]),
            "N_knot": np.arange(self.config["n_knots"] + 2)
        }

        with pm.Model(coords=model_coords) as spline_model:
            
            
            alpha = pm.Normal("alpha", mu=0, sigma=1)

            mu = alpha


            for xcol in self.config["x"]:

                # For penalised spline, we are 
                # saying that the spline weights
                # come from a Normal distribution
                spline_weight_sd = pm.Exponential(
                    f'spline_weight_sd_{xcol}', 1
                )
                spline_weights = pm.Normal(
                    f'spline_weights_{xcol}',
                    mu=0,
                    sigma=spline_weight_sd,
                    dims=('N_knot'),
                )


                basis_matrix = pm.Data(
                    f'{xcol}_basis',
                    self.config["design_matrix"][
                        :, self.config["matrix_indices"][xcol]
                    ].astype("float64"),
                    dims=("N", 'N_knot'),
                )

                mu += pm.math.dot(basis_matrix, spline_weights.T)

            mu = pm.Deterministic(
                "mu",
                mu,
                dims=("N"))
            sigma = pm.Exponential("sigma", 1)



           
            pm.Normal(
                "doy",
                mu=mu,
                sigma=sigma,
                observed=self.data[self.config["y"]],
                dims=("N")
            )
            idata = pm.sample(
                draws=2000,
                tune=1000,
                chains=4,
                target_accept=0.95,
                return_inferencedata=True,
            )

            posterior_predictive = pm.sample_posterior_predictive(
                idata, extend_inferencedata=True, var_names=["mu"]
            ).posterior_predictive

        

        self.model = spline_model
        self.posterior_predictive = posterior_predictive
        self.idata = idata