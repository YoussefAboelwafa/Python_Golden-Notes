# Python Notes
This is my Python Notes Repo
This Repo include Notes for 3 main libraries for Data Science & Machine Learning used in Python
- [numpy](#numpy)
- [matplotlib](#matplotlib)
- [pandas](#pandas)

# numpy
  ## np array boolean



  ```python
  # Import numpy
  import numpy as np
  
  # Calculate the BMI: bmi
  height_in = [74, 74, 72, 72, 73, 69, 69, 71, 76, 71]
  weight_lb = [180, 215, 210, 210, 188, 176, 209, 200, 231, 180]
  
  np_height_m = np.array(height_in) * 0.0254
  np_weight_kg = np.array(weight_lb) * 0.453592
  bmi = np_weight_kg / (np_height_m**2)
  
  # Create the light array (boolean array)
  light = bmi < 25
  print(light)
  # BMIs of all baseball players whose BMI is below 21
  print(bmi[light])
  ```
  
      [ True False False False  True False False False False False]
      [23.11037639 24.80333518]
  
  
  ## Know the number of rows and cols of np array
  
  
  
  ```python
  baseball = [[180, 78.4], [215, 102.7], [210, 98.5], [188, 75.2]]
  
  # Create a 2D numpy array from baseball: np_baseball
  np_baseball = np.array(baseball)
  
  # Print out the type of np_baseball
  print(type(np_baseball))
  
  # Print out the shape of np_baseball
  print(np_baseball.shape)
  ```
  
      <class 'numpy.ndarray'>
      (4, 2)
  
  
  ## 2D Arithmetic
  
  
  
  ```python
  # Import numpy package
  import numpy as np
  
  np_mat = np.array([[1, 2], [3, 4], [5, 6]])
  
  # Create numpy array: conversion
  # Multiply the first col by 2 & the second col by 3
  conversion = np.array([2, 3])
  
  # Print out product of np_baseball and conversion
  print(np_mat * conversion)
  ```
  
      [[ 2  6]
       [ 6 12]
       [10 18]]
  
  
  ## Filtering np array
  
  
  
  ```python
  import numpy as np
  
  positions = ["GK", "M", "A", "D"]
  heights = [191, 184, 185, 180]
  # Convert positions and heights to numpy arrays: np_positions, np_heights
  np_positions = np.array(positions)
  np_heights = np.array(heights)
  
  # Heights of the goalkeepers: gk_heights
  gk_heights = np_heights[np_positions == "GK"]
  
  # Heights of the other players: other_heights
  other_heights = np_heights[np_positions != "GK"]
  
  # Print out the median height of goalkeepers
  print("Median height of goalkeepers: " + str(np.median(gk_heights)))
  
  # Print out the median height of other players
  print("Median height of other players: " + str(np.median(other_heights)))
  ```
  
      Median height of goalkeepers: 191.0
      Median height of other players: 184.0

# matplotlib

## Line Plot

#### Notes

`plt.clf()` to clear



```python
import matplotlib.pyplot as plt

year = [2000, 2010, 2020, 2030, 2040]
pop = [2000, 4000, 1000, 3000, 5000]

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Customization
# ---------------------------------------------

plt.xlabel("Year")
plt.ylabel("Population")
plt.title("World Population Projections")
plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000],[0,'1k','2k','3k','4k','5k','6k'])
plt.text(2010,50,"Text")
plt.grid(True)
plt.xlim(2000,None)
# ---------------------------------------------

# Display the plot with plt.show()
plt.show()
```

## Scatter Plot


#### Note

Use log scale

`plt.xscale('log')`



```python
import matplotlib.pyplot as plt

year = [2000, 2010, 2020, 2030, 2040]
pop = [20, 40, 10, 30, 50]

# Make a line plot: year on the x-axis, pop on the y-axis
plt.scatter(year, pop, color="red", marker="x")

# Display the plot with plt.show()
plt.show()
```

## Histogram

`plt.hist(list,bins)`

bins = 10 by default



```python
import matplotlib.pyplot as plt

x = [
    43.828,
    76.423,
    72.301,
    42.731,
    75.32,
    81.235,
    79.829,
    75.635,
    64.062,
    79.441,
    56.728,
    65.554,
    74.852,
    50.728,
    72.39,
    73.005,
    52.295,
    49.58,
    59.723,
    50.43,
    80.653,
    50.651,
    78.553,
    72.961,
    72.889,
    65.152,
    46.462,

]
plt.hist(x, bins=6, color="red", edgecolor="black", linewidth=2)
plt.show()
```

# pandas

## Dictionary to DataFrame



```python
# Pre-defined lists
names = ["United States", "Australia", "Japan", "India", "Russia", "Morocco", "Egypt"]
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {"country": names, "drives_right": dr, "cars_per_cap": cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Definition of row_labels
row_labels = ["US", "AUS", "JPN", "IN", "RU", "MOR", "EG"]

# Specify row labels of cars
cars.index = row_labels

# Print cars
print(cars)
```

               country  drives_right  cars_per_cap
    US   United States          True           809
    AUS      Australia         False           731
    JPN          Japan         False           588
    IN           India         False            18
    RU          Russia          True           200
    MOR        Morocco          True            70
    EG           Egypt          True            45


## CSV to DataFrame



```python
# Import pandas as pd
import pandas as pd

# Fix import by including index_col
brics = pd.read_csv("datasets/brics.csv")
# Print out cars
print(brics)
print("----------------------------------------------")
# Specify the index_col argument inside pd.read_csv(): set it to 0, so that the first column is used as row labels.
brics2 = pd.read_csv("datasets/brics.csv", index_col=0)
print(brics2)
```

      Unnamed: 0  cars_per_cap       country  drives_right
    0         US           809  UnitedStates          True
    1        AUS           731     Australia         False
    2        JPN           588         Japan         False
    3         IN            18         India         False
    4         RU           200        Russia          True
    5        MOR            70       Morocco          True
    6         EG            45         Egypt          True
    ----------------------------------------------
         cars_per_cap       country  drives_right
    US            809  UnitedStates          True
    AUS           731     Australia         False
    JPN           588         Japan         False
    IN             18         India         False
    RU            200        Russia          True
    MOR            70       Morocco          True
    EG             45         Egypt          True


## Col access using brackets



```python
import pandas as pd

brics = pd.read_csv("datasets/brics.csv", index_col=0)

print(brics)
print("----------------------------------------------")
print(brics[["country", "drives_right"]])
print("----------------------------------------------")
print(brics[["country"]])
print("----------------------------------------------")
```

         cars_per_cap       country  drives_right
    US            809  UnitedStates          True
    AUS           731     Australia         False
    JPN           588         Japan         False
    IN             18         India         False
    RU            200        Russia          True
    MOR            70       Morocco          True
    EG             45         Egypt          True
    ----------------------------------------------
              country  drives_right
    US   UnitedStates          True
    AUS     Australia         False
    JPN         Japan         False
    IN          India         False
    RU         Russia          True
    MOR       Morocco          True
    EG          Egypt          True
    ----------------------------------------------
              country
    US   UnitedStates
    AUS     Australia
    JPN         Japan
    IN          India
    RU         Russia
    MOR       Morocco
    EG          Egypt
    ----------------------------------------------


## Row access using brackets



```python
import pandas as pd

brics = pd.read_csv("datasets/brics.csv", index_col=0)

print(brics)
print("----------------------------------------------")
print(brics[1:4])
print("----------------------------------------------")
```

         cars_per_cap       country  drives_right
    US            809  UnitedStates          True
    AUS           731     Australia         False
    JPN           588         Japan         False
    IN             18         India         False
    RU            200        Russia          True
    MOR            70       Morocco          True
    EG             45         Egypt          True
    ----------------------------------------------
         cars_per_cap    country  drives_right
    AUS           731  Australia         False
    JPN           588      Japan         False
    IN             18      India         False
    ----------------------------------------------


## Rows & Cols access using loc & iloc
```[Subsetting DataFrame]```



```python
import pandas as pd

brics = pd.read_csv("datasets/brics.csv", index_col=0)

print(brics)
print("----------------------------------------------")


# Access row by label
print(brics.loc[["RU"]])
print("----------------------------------------------")
print(brics.iloc[[4]])
print("----------------------------------------------")


# Access multiple rows by label
print(brics.loc[["RU", "IN", "EG"]])
print("----------------------------------------------")
print(brics.iloc[[4, 3, 6]])
print("----------------------------------------------")


# Access row and column by label
print(brics.loc[["RU", "IN", "EG"], ["country", "drives_right"]])
print("----------------------------------------------")
print(brics.iloc[[4, 3, 6], [1, 2]])
print("----------------------------------------------")


# All rows, some columns
print(brics.loc[:, ["country", "drives_right"]])
print("----------------------------------------------")
print(brics.iloc[:, [1, 2]])
print("----------------------------------------------")
```

         cars_per_cap       country  drives_right
    US            809  UnitedStates          True
    AUS           731     Australia         False
    JPN           588         Japan         False
    IN             18         India         False
    RU            200        Russia          True
    MOR            70       Morocco          True
    EG             45         Egypt          True
    ----------------------------------------------
        cars_per_cap country  drives_right
    RU           200  Russia          True
    ----------------------------------------------
        cars_per_cap country  drives_right
    RU           200  Russia          True
    ----------------------------------------------
        cars_per_cap country  drives_right
    RU           200  Russia          True
    IN            18   India         False
    EG            45   Egypt          True
    ----------------------------------------------
        cars_per_cap country  drives_right
    RU           200  Russia          True
    IN            18   India         False
    EG            45   Egypt          True
    ----------------------------------------------
       country  drives_right
    RU  Russia          True
    IN   India         False
    EG   Egypt          True
    ----------------------------------------------
       country  drives_right
    RU  Russia          True
    IN   India         False
    EG   Egypt          True
    ----------------------------------------------
              country  drives_right
    US   UnitedStates          True
    AUS     Australia         False
    JPN         Japan         False
    IN          India         False
    RU         Russia          True
    MOR       Morocco          True
    EG          Egypt          True
    ----------------------------------------------
              country  drives_right
    US   UnitedStates          True
    AUS     Australia         False
    JPN         Japan         False
    IN          India         False
    RU         Russia          True
    MOR       Morocco          True
    EG          Egypt          True
    ----------------------------------------------





```python

```
