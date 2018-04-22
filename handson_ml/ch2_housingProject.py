# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os                          # to interface with the underlying operating system that Python is running on
import tarfile                     # to open compressed .tgz files
from six.moves import urllib

import numpy as np
from sklearn.model_selection import train_test_split
import hashlib

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# to get the data
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # download data directly from website. better like this so it will still work even
    # if data change, or if running from other machines
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")  # download a compressed file
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# multiple options to split train and test set
# important to do that early so to not get biased by the data inspection
    
# use a hash to ensure the same items go to the train and test set every time
def test_set_check(identifier, test_ratio, hash):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]



################################
###### actual analysis #########
################################
    
# retrieve data
fetch_housing_data()
housing = load_housing_data()

# explore the data
print(housing.head()) # when running script I need a print call to visualize, not sure why
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

# explore the data - graphs
housing.hist(bins=50, figsize=(10,7))
# housing["median_house_value"].hist(bins=50, figsize=(10,7))
plt.show()
#save_fig("attribute_histogram_plots")

###### split train and test set 

# with hash
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
print(test_set.head())

# alternatively, use predefined function, 
# setting the rundom state ensure the same test and training sets are selected every time, 
# but only if new eventual data points are added to the dataset at the bottom
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(test_set.head())

# when splitting the dataset, consider stratified sampling rather than normal sampling as above
# for instance if you look at medianincome it is clear that chances are higher to sample around the median
# this is good, but you also want to ensure that more distant values are also represented
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5) # Divide by 1.5 to limit the number of income categories
# not sure this actually makes sense, why 1.5 and not 1, to keep the intuitive meaning and order of magnitude
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True) # Label those above 5 as 5
housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# now check that ratios are respected
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(housing["income_cat"].value_counts() / len(housing))
# now resplit data with startified sampling
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print(compare_props)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


