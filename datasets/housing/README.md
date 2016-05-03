# California Housing

## Source
This dataset is a modified version of the California Housing dataset available from [http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html](Luís Torgo's page) (University of Porto). Luís Torgo obtained it from the StatLib repository (which is closed now). The dataset may also be downloaded from StatLib mirrors.

This dataset appeared in a 1997 paper titled *Sparse Spatial Autoregressions* by Pace, R. Kelley and Ronald Barry, published in the *Statistics and Probability Letters* journal. They built it using the 1990 California census data. It contains one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

## Tweaks
The dataset in this directory is almost identical to the original, with two differences:

* 207 values were randomly removed from the `total_bedrooms` column, so we can discuss what to do with missing data.
* An additional categorical attribute called `ocean_proximity` was added, indicating (very roughly) whether each block group is near the ocean, near the Bay area, inland or on an island. This allows discussing what to do with categorical data.

Note that the block groups are called "districts" in the Jupyter notebooks, simply because in some contexts the name "block group" was confusing.

## Data description

    >>> housing.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
    longitude             20640 non-null float64
    latitude              20640 non-null float64
    housing_median_age    20640 non-null float64
    total_rooms           20640 non-null float64
    total_bedrooms        20433 non-null float64
    population            20640 non-null float64
    households            20640 non-null float64
    median_income         20640 non-null float64
    median_house_value    20640 non-null float64
    ocean_proximity       20640 non-null object
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    
    >>> housing["ocean_proximity"].value_counts()
    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64
    
    >>> housing.describe()
              longitude      latitude  housing_median_age   total_rooms  \
    count  16513.000000  16513.000000        16513.000000  16513.000000   
    mean    -119.575972     35.639693           28.652335   2622.347605   
    std        2.002048      2.138279           12.576306   2138.559393   
    min     -124.350000     32.540000            1.000000      6.000000   
    25%     -121.800000     33.940000           18.000000   1442.000000   
    50%     -118.510000     34.260000           29.000000   2119.000000   
    75%     -118.010000     37.720000           37.000000   3141.000000   
    max     -114.310000     41.950000           52.000000  39320.000000   

           total_bedrooms    population    households  median_income  
    count    16355.000000  16513.000000  16513.000000   16513.000000  
    mean       534.885112   1419.525465    496.975050       3.875651  
    std        412.716467   1115.715084    375.737945       1.905088  
    min          2.000000      3.000000      2.000000       0.499900  
    25%        295.000000    784.000000    278.000000       2.566800  
    50%        433.000000   1164.000000    408.000000       3.541400  
    75%        644.000000   1718.000000    602.000000       4.745000  
    max       6210.000000  35682.000000   5358.000000      15.000100
 