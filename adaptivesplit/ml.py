import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, \
    explained_variance_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection._split import check_cv

_njobs_ = 1
_def_regression_score = "neg_mean_squared_error"
_def_classification_score = "accuracy"


def train(X, y, model, p_grid, nested=False, score=None,
          inner_cv=LeaveOneOut(), outer_cv=LeaveOneOut()):
    """
    Train the model, including model selection, model finalization and possibly nested cross-validation.
    Printing some useful info...
    :param X: features
    :param y: target variable
    :param model: a scikit-learn estimator
    :param p_grid: hyperparameter grid for cross-validation
    :param nested: do nested cross validation
    :param score: scikit learn scoring metric
    :param inner_cv: scikitlearn-line cross validation specification
    :param outer_cv: scikitlearn-line cross validation specification (if nested is True)
    :return: the winning model, finalized by fitting to the whole data
    """
    if score is None:
        if is_regressor(model):
            score = _def_regression_score
        else:
            score = _def_classification_score

    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring=score, verbose=False,
                       return_train_score=False, n_jobs=_njobs_)
    clf.fit(X, y)

    print("**** Non-nested analysis ****")
    print("* Best hyperparameters: " + str(clf.best_params_))
    print("* Scorer: ", score)
    if is_regressor(model):
        print("* Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X),
                                                                                  y_true=y)))
        print("* Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("* Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))
        print("** Explained Variance: " + str(
            1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))

    elif is_classifier(model):
        print("* Accuracy on full data as training set: " + str(accuracy_score(y_pred=clf.best_estimator_.predict(X),
                                                                               y_true=y)))
        print("* Best Non-nested cross-validated score on test: " + str(clf.best_score_))
        print("** Precision: " + str(precision_score(y_pred=clf.best_estimator_.predict(X),
                                                     y_true=y)))
        print("** Recall: " + str(recall_score(y_pred=clf.best_estimator_.predict(X),
                                               y_true=y)))

    model = clf.best_estimator_

    avg_model = None
    all_models = []
    if nested:
        print("**** Nested cross-validation analysis ****")

        best_params = []
        predicted = np.zeros(len(y))
        actual = np.zeros(len(y))
        nested_scores_train = np.zeros(outer_cv.get_n_splits(X))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations 'manually', for more control
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y):
            clf.fit(X[train], y[train])

            all_models.append(clf.best_estimator_)

            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted[i] = clf.predict(X[test])
            actual[i] = y[test]

            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])
            i = i + 1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner cross-validation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Cross-validation Score (outer_cv):\t" + str(nested_scores_test.mean()))

        print("Explained Variance: " + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(),
                                                                                                         len(y)), y)))
        print("Correlation: " + str(np.corrcoef(actual, predicted)[0, 1]))

        # plot the prediction of the outer cv
        fig, ax = plt.subplots()
        ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()],
                [y.min(), y.max()],
                'k--', lw=2)
        ax.set_xlabel('Pain Sensitivity')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title("Expl. Var.:" +
                  str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
                  "\nCorrelation: " + str(np.corrcoef(actual, predicted)[0, 1]))
        plt.show()
    else:
        all_models = [model]

    model.fit(X, y)  # fit to whole data

    return model


def train_kb(X, y, model, p_grid, nested=False, score="neg_mean_squared_error",
          inner_cv=LeaveOneOut(), outer_cv=LeaveOneOut()):
    """
    Train the model, including model selection, model finalization and possibly nested cross-validation.
    Printing some useful info...

    :param X: features
    :param y: target variable
    :param model: a scikit-learn estimator
    :param p_grid: hyperparameter grid for cross-validation
    :param nested: do nested cross validation
    :param score: scikit learn scoring metric
    :param inner_cv: scikitlearn-line cross validation specification
    :param outer_cv: scikitlearn-line cross validation specification (if nested is True)
    :return: the winning model, finalized by fitting to the whole data
    """
    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring=score, verbose=False,
                       return_train_score=False, n_jobs=_njobs_)
    clf.fit(X, y)
    maki = clf.cv_results_

    print("**** Non-nested analysis ****")
    print("* Best hyperparameters: " + str(clf.best_params_))

    print("* Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X),
                                                                              y_true=y)))
    print("* Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    print("* Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))

    model = clf.best_estimator_

    print("** Explained Variance: " + str(
        1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))

    avg_model = None
    all_models = []
    if nested:
        print("**** Nested cross-validation analysis ****")

        best_params = []
        predicted = np.zeros(len(y))
        actual = np.zeros(len(y))
        nested_scores_train = np.zeros(outer_cv.get_n_splits(X))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations 'manually', for more control
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y):
            clf.fit(X[train], y[train])

            all_models.append(clf.best_estimator_)

            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted[i] = clf.predict(X[test])
            actual[i] = y[test]

            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])
            i = i+1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner cross-validation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Cross-validation Score (outer_cv):\t" + str(nested_scores_test.mean()))

        print("Explained Variance: " + str(1 - nested_scores_test.mean()/-mean_squared_error(np.repeat(y.mean(),
                                                                                                       len(y)), y)))
        print("Correlation: " + str(np.corrcoef(actual, predicted)[0, 1]))

        #plot the prediction of the outer cv
        fig, ax = plt.subplots()
        ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()],
                [y.min(), y.max()],
                'k--', lw=2)
        ax.set_xlabel('Pain Sensitivity')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title("Expl. Var.:" +
                  str(1 - nested_scores_test.mean()/-mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
                  "\nCorrelation: " + str(np.corrcoef(actual, predicted)[0, 1]))
        plt.show()
    else:
        all_models = [model]

    model.fit(X, y)  # fit to whole data

    return model, maki

def pred_stat(observed, predicted, robust=False):
    """
    Simple parametric analysis of the prediction strength.
    ToDo: CAUTION: permutation-test based statistics SHOULD be used for more appropriate results.
    :param observed: observed values of the target variable
    :param predicted: predicted values
    :param robust: whether to do robust regression (in case of outliers)
    :return: p-value, R-squared, residual data, regression line (for plotting)
    """
    # convert to np.array
    observed = np.array(observed)
    predicted = np.array(predicted)

    # EXCLUDE NA-s:
    predicted = predicted[~np.isnan(observed)]
    observed = observed[~np.isnan(observed)]

    if robust:
        res = sm.RLM(observed, sm.add_constant(predicted)).fit()
        p_value = res.pvalues[1]
        regline = res.fittedvalues
        residual = res.sresid

        # this is a pseudo r_squared, see: https://stackoverflow.com/questions/31655196/how-to-get-r-squared-for-robust-regression-rlm-in-statsmodels
        r_2 = sm.WLS(observed, sm.add_constant(predicted), weights=res.weights).fit().rsquared

    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
        regline = slope * observed + intercept
        r_2 = r_value ** 2
        residual = observed - regline

    return p_value, r_2, residual, regline


def plot_prediction(observed, predicted, outfile="", covar=[], robust=False, sd=True, text=""):
    """
    Predicted-observed plot.
    :param observed: observed values of the target variable
    :param predicted: predicted values
    :param outfile: file to write plot into (default: no output file)
    :param covar: I am not sure anymore, I guess you can color the points by a covariate of choice.
    :param robust: whether to do robust regression (in case of outliers)
    :param sd: whether to plot standard deviation
    :param text: title text for the plot
    """
    color = "black"
    if len(covar):
        g = sns.jointplot(observed, predicted, scatter=False, color=color, kind="reg", robust=robust, x_ci="sd", )
        plt.scatter(observed, predicted,
                    c=covar, cmap=ListedColormap(sns.color_palette(["#5B5BFF", "#D73E68"])))
    else:
        g = sns.jointplot(observed, predicted, kind="reg", color=color, robust=robust, x_ci="sd")

    if sd:
        xlims = np.array(g.ax_joint.get_xlim())
        if robust:
            res = sm.RLM(predicted, sm.add_constant(observed)).fit()
            coefs = res.params
            residual = res.resid
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
            coefs = [intercept, slope]
            regline = slope * observed + intercept
            residual = observed - regline

        S = np.sqrt(np.mean(residual ** 2))
        upper = coefs[1] * xlims + coefs[0] + S / 2
        lower = coefs[1] * xlims + coefs[0] - S / 2

        plt.plot(xlims, upper, ':', color=color, linewidth=1, alpha=0.3)
        plt.plot(xlims, lower, ':', color=color, linewidth=1, alpha=0.3)

    if text:
        plt.text(np.min(observed) - (np.max(predicted) - np.min(predicted)) / 3,
                 np.max(predicted) + (np.max(predicted) - np.min(predicted)) / 3,
                 text, fontsize=10)

    if outfile:
        figure = plt.gcf()
        figure.savefig(outfile, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()


# for external validation only!
def evaluate_prediction(model, X, y, orig_mean=None, outfile="", robust=False, covar=[]):
    """
    Evaluate prediction by calculating various metrics and plotting a nice predicted-observed plot.
    This is *NOT* cross-validated prediction, use it only for external validation!
    :param model: a scikit-learn estimator
    :param X: features
    :param y: target variable
    :param orig_mean: mean of the training sample (to calculate "fair" explained variance in external validation)
    :param outfile: file to write plot into (default: no output file)
    :param robust: whether to do robust regression (in case of outliers)
    :param covar: I am not sure anymore, I guess you can color the points by a covariate of choice.
    :return: predicted values
    """

    predicted = model.predict(X)

    if is_regressor(model):
        p_value, r_2, residual, regline = pred_stat(y, predicted, robust=robust)

        if orig_mean:
            y_base = orig_mean
        else:
            y_base = y.mean()

        expl_var = (1 - (-mean_squared_error(y_pred=predicted, y_true=y)
                         /
                         -mean_squared_error(np.repeat(y_base, len(y)), y))) * 100

        print("R2=" + "{:.3f}".format(r_2) + "  R=" + "{:.3f}".format(np.sqrt(r_2)) + '\n'
              + "   p=" + "{:.6f}".format(p_value) + "  Expl. Var.: " + "{:.1f}".format(expl_var) + "%" + '\n'
              + "  Expl. Var.2: " + "{:.1f}".format(
            explained_variance_score(y_pred=predicted, y_true=y) * 100) + "%" + '\n'
              + "  MSE=" + "{:.3f}".format(mean_squared_error(y_pred=predicted, y_true=y)) + '\n'
              + " RMSE=" + "{:.3f}".format(np.sqrt(mean_squared_error(y_pred=predicted, y_true=y))) + '\n'
              + "  MAE=" + "{:.3f}".format(mean_absolute_error(y_pred=predicted, y_true=y)) + '\n'
              + " MedAE=" + "{:.3f}".format(median_absolute_error(y_pred=predicted, y_true=y)) + '\n'
              + "  R^2=" + "{:.3f}".format(r2_score(y_pred=predicted, y_true=y))) + '\n'

        plot_prediction(y, predicted, outfile, robust=robust, sd=True, covar=covar,
                        text="$R^2$ = " + "{:.3f}".format(r_2) +
                             "  p = " + "{:.3f}".format(p_value) +
                             " Expl. Var.: " + "{:.1f}".format(expl_var)
                        )

    elif is_classifier(model):
        print("ACC=" "{:.3f}".format(accuracy_score(y, predicted)) + '\n' +
              "Recall=" "{:.3f}".format(recall_score(y, predicted)) + '\n' +
              "Precision=" "{:.3f}".format(precision_score(y, predicted)) + '\n' +
              "ROC AUC=" "{:.3f}".format(roc_auc_score(y, predicted)))

        conf_mat = confusion_matrix(y_true=y, y_pred=predicted)

        sns.heatmap(conf_mat, annot=True, cmap='Blues')
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")

        print("** Clasification Report **")
        print(classification_report(y_true=y, y_pred=predicted))
    return predicted


def evaluate_crossval_prediction(model, X, y, outfile="", cv=LeaveOneOut(), robust=False):
    """
    Evaluate prediction by calculating various metrics and plotting a nice predicted-observed plot.
    This *is* cross-validated prediction, re-fits the model, this should be used on the training sample!
    :param model: a scikit-learn estimator
    :param X: features
    :param y: target variable
    :param orig_mean: mean of the training sample (to calculate "fair" explained variance in external validation)
    :param outfile: file to write plot into (default: no output file)
    :param robust: whether to do robust regression (in case of outliers)
    :return: cross-validated predictions
    """
    predicted = cross_val_predict(model, X, y, cv=cv, n_jobs=_njobs_)

    if is_regressor(model):
        p_value, r_2, residual, regline = pred_stat(y, predicted, robust=robust)

        expl_var = (1 - (-mean_squared_error(y_pred=predicted, y_true=y)
                         /
                         -mean_squared_error(np.repeat(y.mean(), len(y)), y))) * 100

        print("R2=" + "{:.3f}".format(r_2) + "  R=" + "{:.3f}".format(np.sqrt(r_2)) + '\n'
              + "p=" + "{:.6f}".format(p_value) + "  Expl. Var.: " + "{:.1f}".format(expl_var) + "%" + '\n'
              + "Expl. Var.2: " + "{:.1f}".format(
            explained_variance_score(y_pred=predicted, y_true=y) * 100) + "%" + '\n'
              + "MSE=" + "{:.3f}".format(mean_squared_error(y_pred=predicted, y_true=y)) + '\n'
              + "RMSE=" + "{:.3f}".format(np.sqrt(mean_squared_error(y_pred=predicted, y_true=y))) + '\n'
              + "MAE=" + "{:.3f}".format(mean_absolute_error(y_pred=predicted, y_true=y)) + '\n'
              + "MedAE=" + "{:.3f}".format(median_absolute_error(y_pred=predicted, y_true=y)) + '\n'
              + "R^2=" + "{:.3f}".format(r2_score(y_pred=predicted, y_true=y)))

        plot_prediction(y, predicted, outfile, robust=robust, sd=True,
                        text="$R2$=" + "{:.3f}".format(r_2) +
                             "  p=" + "{:.3f}".format(p_value) +
                             "  Expl. Var.: " + "{:.1f}".format(expl_var) + "%"
                        )
    elif is_classifier(model):
        print("ACC=" "{:.3f}".format(accuracy_score(y, predicted)) + '\n' +
              "Recall=" "{:.3f}".format(recall_score(y, predicted)) + '\n' +
              "Precision=" "{:.3f}".format(precision_score(y, predicted)) + '\n' +
              "ROC AUC=" "{:.3f}".format(roc_auc_score(y, predicted)))

        conf_mat = confusion_matrix(y_true=y, y_pred=predicted)

        sns.heatmap(conf_mat, annot=True, cmap='Blues')
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")

        print("** Clasification Report **")
        print(classification_report(y_true=y, y_pred=predicted))
    return predicted


def stderr(data):
    """
    Standard error.
    :param data: data
    :return: standard error
    """
    return np.std(data, ddof=1) / np.sqrt(np.size(data))


def ci_stderr(data):
    """
    Helper function for plotting standard-error.
    :param data: data
    :return: upper, lower
    """
    return np.mean(data) - stderr(data) / 2, np.mean(data) + stderr(data) / 2


def ci_95(data):
    """
    Helper function for plotting 95% confidence intervals.
    :param data: data
    :return: upper, lower
    """
    return np.percentile(data, [2.5, 97.5])


def learning_curve(model, df, yname, xnames, Ns, cv, shuffle=10,
                   cv_statfun=np.mean, statfun=np.mean, ci_fun=ci_stderr):
    """
    Calculate learning curve.
    You can plot it e.g. like this:
    import seaborn as sns
    sns.lineplot(Ns, train,color="blue")
    sns.lineplot(Ns, test, color="red")
    sns.lineplot(Ns, ci_train_lower, color="blue", linestyle=':')
    sns.lineplot(Ns, ci_train_upper, color="blue", linestyle=':')
    sns.lineplot(Ns, ci_test_lower, color="red", linestyle=':')
    sns.lineplot(Ns, ci_test_upper, color="red", linestyle=':')
    :param model: a scikit-learn estimator
    :param df: pandas data-frame containing both the target and the features
    :param yname: name of the columns containing the target variable
    :param xnames: name of the feature columns
    :param Ns: list of sample sizes used for learning-curev calculation
    :param cv: cross-validation a'la scikit-learn
    :param shuffle: number of shuffles, for more stable learning-curve estimates
    :param cv_statfun: stat function for aggregating the the cross-validated results
    :param statfun: statfun to aggregate across shuffles
    :param ci_fun: function for the computation of confidence intervals
    :return: all data for plotting, see example above.
    """
    train = []
    test = []

    ci_train_upper = []
    ci_train_lower = []
    ci_test_upper = []
    ci_test_lower = []


    if is_regressor(model):
        score = _def_regression_score
    else:
        score = _def_classification_score

    for n in Ns:
        print("******************")
        print(n)

        tr = []
        te = []
        for s in range(shuffle):
            sample = df.sample(n)

            cvfit = cross_validate(model, sample[xnames], sample[yname], scoring=score,
                                   cv=cv, return_train_score=True, n_jobs=_njobs_)

            ##
            tr.append(cv_statfun(cvfit["train_score"]))
            te.append(cv_statfun(cvfit["test_score"]))

        print(statfun(tr), statfun(te))

        ci_train = ci_fun(tr)
        ci_train_lower.append(ci_train[0])
        ci_train_upper.append(ci_train[1])

        ci_test = ci_fun(te)
        ci_test_lower.append(ci_test[0])
        ci_test_upper.append(ci_test[1])

        train.append(statfun(tr))
        test.append(statfun(te))

    return train, test, ci_train_lower, ci_train_upper, ci_test_lower, ci_test_upper


def power_corr(corr, N, iters=1000, alpha=0.05, plot=False):
    """
    Calculate and (optionally plot) power for a given correlation, sample size and alpha threshold.
    Based on Monte-Carlo simulations.
    :param corr: assumed correlation
    :param N: number of subjects
    :param iters: number of iterations for the simulation
    :param alpha: rejections threshold
    :param plot: create plot
    :return: estimated power
    """
    rs = []
    ps = []
    for i in range(iters):
        data = np.random.multivariate_normal([0, 0], [[1, corr], [corr, 1]], N)
        # test
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[:, 0],
                                                                       data[:, 1])
        rs.append(r_value)
        ps.append(p_value)

    if plot:
        sns.distplot(rs)
        plt.show()
        sns.distplot(ps, kde=False)
        plt.axvline(alpha, 0, 1)
        plt.show()

    power = np.sum(np.array(ps) < alpha) / iters
    return power


def power_curve(model, df, yname, xnames, Ns, cv, shuffle=10, statfun=np.mean, ci_fun=ci_stderr, power_iters=1000):
    """
    Calculate and plot power curve.
    :param model: a scikit-learn estimator
    :param df: pandas data-frame containing both the target and the features
    :param yname: name of the columns containing the target variable
    :param xnames: name of the feature columns
    :param Ns: list of sample sizes used for learning-curev calculation
    :param cv: cross-validation a'la scikit-learn
    :param shuffle: number of shuffles, for more stable learning-curve estimates
    :param statfun: statfun to aggregate across shuffles
    :param ci_fun: function for the computation of confidence intervals
    :param power_iters:
    :return: na=number of power-simulations
    """
    corr_test = []
    corr_upper = []
    corr_lower = []
    power_test = []
    power_upper = []
    power_lower = []

    for n in Ns:
        print("******************")
        print(n)

        tr = []
        te = []
        corr_test_inner = []
        for s in range(shuffle):
            sample = df.sample(n)

            predictions = cross_val_predict(model, sample[xnames], sample[yname], cv=cv, n_jobs=_njobs_)
            corr_test_inner.append(np.corrcoef((sample[yname], predictions))[0, 1])

        r = statfun(corr_test_inner)
        corr_test.append(r)
        ci_r = ci_fun(corr_test_inner)
        corr_lower.append(ci_r[0])
        corr_upper.append(ci_r[1])

        if n == max(Ns):
            power_test.append(0)
            power_upper.append(0)
            power_lower.append(0)
        else:
            power_test.append(power_corr(corr=r,
                                         N=max(Ns) - n,
                                         iters=power_iters,
                                         alpha=0.05,
                                         plot=False))
            power_upper.append(power_corr(corr=ci_r[1],
                                          N=max(Ns) - n,
                                          iters=power_iters,
                                          alpha=0.05,
                                          plot=False))
            power_lower.append(power_corr(corr=ci_r[0],
                                          N=max(Ns) - n,
                                          iters=power_iters,
                                          alpha=0.05,
                                          plot=False))

        print(statfun(corr_test_inner), power_test[-1])

    # Learning curve, correlation
    sns.lineplot(Ns, corr_test, color="red")
    sns.lineplot(Ns, corr_upper, color="red", linestyle=':')
    sns.lineplot(Ns, corr_lower, color="red", linestyle=':')
    sns.lineplot(Ns, power_test, color="blue")
    sns.lineplot(Ns, power_upper, color="blue", linestyle=':')
    sns.lineplot(Ns, power_lower, color="blue", linestyle=':')
    plt.show()

    return corr_test, power_test
