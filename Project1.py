import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import Project1Constants as const
from sklearn.metrics import mean_squared_error


def loadData(filePath):

    """Load the given matlab data file into the memory.
    Parameters
    ----------
    filePath : The load file.

    Returns
    -------
    Returns dictionary of the data
    """

    return scipy.io.loadmat(filePath)

def getDateBasedOnKey(dataSet, key):

    """Fetch the rows from a data set based on the keys.
    Parameters
    ----------
    dateSet : Dictionary.
    Key : Key

    Returns
    -------
    Returns values as a  column of a given key
    """

    return dataSet[key]

def getFeaturesAndTargets(data):

    """Fetch features and target columns from the given data.
    Parameters
    ----------
    date : {array-like, matrix} Samples.

    Returns
    -------
    data : Returns a column of features and a column of targets
    """

    return data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)

def buildPipeline(degree, model):

    """Build a pipeline of given features and model.
    Parameters
    ----------
    degree : Number of polynomial degrees.
    model : Instance of model type

    Returns
    -------
    Returns a pipeline of model type and given polynomial degrees
    """

    return Pipeline([('poly', PolynomialFeatures(degree=degree)), ('initializedModel', model)])

def plotGraph(x, y, predictions, title, xLabel, yLabel):

    """Build a pipeline of given features and model.
    Parameters
    ----------
    degree : Number of polynomial degrees.
    model : Instance of model type

    Returns
    -------
    Returns a pipeline of model type and given polynomial degrees
    """

    # Fill graph with features and targets
    plt.scatter(x, y);

    # We don't need predictions in experiment# 2, hence checking.
    if predictions.size > 0:
        # Plot predictions on input features
        plt.plot(x, predictions, color='blue', linewidth=3)

    plt.title(title)    # Set the graph title
    plt.xlabel(xLabel)  # Set the label for feature inputs
    plt.ylabel(yLabel)  # set the label for target outputs

    plt.show()          # Render the graph


def model(x, y, model, leaveOneOut):

    """ Creates a model of given parameters.
     Parameters
     ----------
     x : Input features
     y : Target values
     model : instance of the model
     leaveOneOut : True or False - States weather you want to predict using LeaveOneOut or not

     Returns
     -------
     predictions : returns the array of predicted values
     """

    if leaveOneOut:  # if we want to predict using leave one out.
        # create temporary array of shape n*1 to store prediction calculated by LOOCV
        predictions_temp = np.empty(shape=[0, 1])
        # initialize leave one out
        leaveOneOut = LeaveOneOut();
        # split the input features - it will give us the number of split generations.
        leaveOneOut.get_n_splits(x)
        for train_index, test_index in leaveOneOut.split(x):
            # fit the data to given model
            model.fit(x[train_index], y[train_index])
            #predict using split one and store all the predictions to temp array
            predictions_temp = np.append(predictions_temp, model.predict(x[test_index]), axis=0)
            # copy the prediction on prediction as we are returning this later
            predictions = predictions_temp;
    else:  # if we don't want to predict using leave one out.
        # fit the given model
        model.fit(x, y)
        # The predict method predict the values of x based on linear equation : y = mx + c
        predictions = model.predict(x)

    return predictions



if __name__ == "__main__":

    # Experiment 1
    olympicsData = loadData('olympics.mat')


    # Experiment 2
    male100_2 = getDateBasedOnKey(olympicsData, 'male100')
    x, y = getFeaturesAndTargets(male100_2)
    plotGraph(x, y, np.array([]), const.EXPERIMENT_2_TITLE, const.EXPERIMENT_2_X_LABEL, const.EXPERIMENT_2_Y_LABEL)


    # Experiment 3
    '''
    Result - The predictions are almost same as compared to those found in section 1.2.
    For year 2012 - 9.59471385
    For year 2016 - 9.54139031
    '''
    male100_3 = getDateBasedOnKey(olympicsData, 'male100')
    x, y = getFeaturesAndTargets(male100_3)
    modelMale100_3 = LinearRegression()
    modelMale100_3.fit(x, y)
    print('Experiment 3 - Coefficient is: ', modelMale100_3.coef_)
    print('Experiment 3 - Prediction for 2012 is: ',modelMale100_3.predict(2012))
    print('Experiment 3 - Prediction for 2016 is: ',modelMale100_3.predict(2016))


    # Experiment 4
    '''
    Result - Plotted the graph and reproduced the figure 1.5 in the textbook.
    '''
    male100_4 = getDateBasedOnKey(olympicsData, 'male100')
    x, y = getFeaturesAndTargets(male100_4)
    modelMale100_4 = LinearRegression()
    prediction = model(x, y, modelMale100_4, False)
    error = mean_squared_error(y, prediction)
    print('Experiment 4 - Male 100 Error: ', error)
    plotGraph(x, y, prediction, const.EXPERIMENT_4_TITLE, const.EXPERIMENT_4_X_LABEL, const.EXPERIMENT_4_Y_LABEL)


    # Experiment 5
    '''
    Result - The error in male 100 was lower than the error in female 400.
    Mean Squared Error for male100 - 0.0503071104757
    Mean Squared Error for female400 - 0.848981206294
    '''
    female400_5 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_5)
    modelFemale400_5 = LinearRegression()
    prediction = model(x, y, modelFemale400_5, False)
    error = mean_squared_error(y, prediction)
    print('Experiment 5 - Female 400 Error: ', error)
    plotGraph(x, y, prediction, const.EXPERIMENT_5_TITLE, const.EXPERIMENT_5_X_LABEL, const.EXPERIMENT_5_Y_LABEL)


    # Experiment 6
    '''
    Result - The error improves significantly with 3rd degree polynomial as compared to simple linear model.
    Mean Squared Error for female400 with 3rd degree polynomial - 0.134244240943
    '''
    female400_6 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_6)
    modelFemale400_6 = buildPipeline(3, LinearRegression())
    prediction = model(x, y, modelFemale400_6, False)
    error = mean_squared_error(y, prediction)
    print('Experiment 6 - 3rd Degree Polynomial Error: ', error)
    plotGraph(x, y, prediction, const.EXPERIMENT_6_TITLE, const.EXPERIMENT_6_X_LABEL, const.EXPERIMENT_6_Y_LABEL)


    # Experiment 7
    '''
    Result - The error with 5th degree polynomial has very slight improvement as compared to 3rd degree polynomial.
    Mean Squared Error for female400 with 5th degree polynomial - 0.134210747781
    '''
    female400_7 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_7)
    modelFemale400_7 = buildPipeline(5, LinearRegression())
    prediction = model(x, y, modelFemale400_7, False)
    error = mean_squared_error(y, prediction)
    print('Experiment 7 - 5th Degree Polynomial Error: ', error)
    plotGraph(x, y, prediction, const.EXPERIMENT_7_TITLE, const.EXPERIMENT_7_X_LABEL, const.EXPERIMENT_7_Y_LABEL)


    # Experiment 8.1 - 3rd degree leave one out
    '''
    Result -
    Mean Squared Error for female400 with 3rd degree polynomial with LeaveOneOut - 0.562443203374
    '''
    female400_8_1 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_8_1)
    leaveOneOut = LeaveOneOut();
    leaveOneOut.get_n_splits(x)
    modelFemale400_8_1 = buildPipeline(3, LinearRegression())
    prediction = model(x, y, modelFemale400_8_1, True)
    error = mean_squared_error(y, prediction)
    print('Experiment 8.1 - LeaveOneOut 3rd Degree Polynomial Error: ', error)
    plotGraph(x, y, prediction, const.EXPERIMENT_8_1_TITLE, const.EXPERIMENT_8_1_X_LABEL, const.EXPERIMENT_8_1_Y_LABEL)


    # Experiment 8.2 - 5th degree leave one out
    '''
    Result - 5th degree polynomial has a slight improvement over 3rd degree wit LeaveOneOut.
    Mean Squared Error for female400 with 5th degree polynomial with LeaveOneOut - 0.557527701465
    '''
    female400_8_2 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_8_2)
    modelFemale400_8_2 = buildPipeline(5, LinearRegression())
    prediction = model(x, y, modelFemale400_8_2, True)
    error = mean_squared_error(y, prediction)
    print('Experiment 8.2 - LeaveOneOut 5th Degree Polynomial Error: ', error)
    plotGraph(x, y, prediction, const.EXPERIMENT_8_2_TITLE, const.EXPERIMENT_8_2_X_LABEL, const.EXPERIMENT_8_2_Y_LABEL)


    # Experiment 9 Ridge
    '''
    Result - The linear model was better fit in my case compared to Ridge. 
    '''
    female400_9 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_9)
    modelFemale400_9 = buildPipeline(5, Ridge(alpha=0.1))
    prediction = model(x, y, modelFemale400_9, False)
    print('Experiment 9 Ridge - Coefficient is: ', modelFemale400_9.named_steps['initializedModel'].coef_)
    plotGraph(x, y, prediction, const.EXPERIMENT_9_TITLE, const.EXPERIMENT_9_X_LABEL, const.EXPERIMENT_9_Y_LABEL)


    # Experiment 10 - Ridge with many alpha values
    alpha_ridge = [0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0]
    female400_9 = getDateBasedOnKey(olympicsData, 'female400')
    x, y = getFeaturesAndTargets(female400_9)
    for a in alpha_ridge:
        modelFemale400_9 = buildPipeline(5, Ridge(alpha=a))
        prediction = model(x, y, modelFemale400_9, False)
        plotGraph(x, y, prediction, const.EXPERIMENT_10_TITLE + str(a), const.EXPERIMENT_10_X_LABEL, const.EXPERIMENT_10_Y_LABEL)
