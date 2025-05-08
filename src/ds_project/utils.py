import numpy as np
import pandas as pd

def generate_test_data(seed=123):
    """
    Generating a test dataset. In this test 
    dataset, the number of days in bloom is 
    related to temperature, temperature also 

    """
    year_min = 800
    year_max = 2025
    years = np.arange(year_min, year_max)

    # Let temperature begin to increase from 1800s
    # and then oscillate around a trend
    temp = []
    for year in years:
        if year < 1800:
            temp.append(np.random.normal(5, 2.5))
        if year >=1800:
            random_temp = np.random.normal(5, 2.5)
            random_temp += 0.01 * (year - 1800)**1.5 
            temp.append(random_temp)
    
    temp = np.array(temp)
    
    doy = 100 + \
    5*np.sin(0.0075*(years - year_min)) - \
    2*temp*np.sin(0.01*temp) + \
    np.random.normal(0, 2, len(years))

    mean_temp = np.mean(temp)
    mean_year = np.mean(years)

    conditional_year_effect = 100 + \
    5*np.sin(0.0075*(years - year_min)) - \
    2*mean_temp*np.sin(0.01*mean_temp)

    conditional_temp_effect = 100 + \
    5*np.sin(0.0075*(mean_year - year_min)) - \
    2*temp*np.sin(0.01*temp)

    df = pd.DataFrame({'year': years, 
                       'temp': temp, 
                       'doy': doy,
                       'conditional_year_effect': conditional_year_effect,
                       'conditional_temp_effect': conditional_temp_effect})

    return(df)