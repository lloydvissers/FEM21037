# FEM21037
Duplicate search

This README file will be a guidance through the python code.
# Main functions
## Function: run()
This function will start the script.
Within this function, multiple parameters can be changed

Parameters
k_range, this is the value represents the amount of bootstraps taken to get more reliable estimates.
r_range, here a list can be provided of all the r values you want to loop through. In the final output this is displayed
t_range, a list, can be provided for a range of thresholds used in the jaccard similarity measure.

The results are obtained via a multicore processing tool, in order to speed up the calculation time. Here the worker function is called multiple times, such that all combination of parameter are calculated. The results of all combinations are returned.

Next a data frame is made where all the results are converted to a nice table.
And below there is a plot function, plotting some of the results.

## Function: worker(k,r,t)
The input for the worker are equal to the predefined parameters.

First, the data is imported from a csv file. Note that the provided json-file is convert to a csv file, because this was handier to use.
Second, a column is added to the dataset where the brand of each product is represented in a separate column. 

Then the dataset is split in two part, one is the training dataset and two is the test dataset, at a fraction of 60% and 40% respectively.

For both the train and test dataset, the “execute_duplicatesearch” functions are called.
Finely, the results are returned including their parameter settings.

## Function: execute_duplicatesearch(data, r, t)
In this function, the complete whole duplicate search algorithm is run.

data_c: represents a list with all model words for each product.
binary_vector_matrix: the binary vector matrix is generated
n = set equal to half of the length of the binary vector matrix. Rounded to the next 10 value.
signature_matrix: is generated by applying the min-hash function.
bands: the signature matrix is divided into b bands. Where b = n/r
candidatepairs: are obtained by applying the LSH function

pair_estimates: estimated to be a pair if the candidate pairs similarity is above threshold t
pair_real: all known real pairs calculated based on their modelID

# Most important specific functions
## Function: get_clean_data(data)
Here, the data used in cleaned.
Only the “title” column is used.

First two lists are provided for the possible writings of ‘inch’ and ‘hertz’. 
For every title in the dataset, a list of words in the title is made. Only added to the list if there are numbers and or text in the word.

Then every “inch” and “hertz” are standardized. A final check is made to ensure that all model words contain a number and character.

## Function: map_binary_1(mw_all, mw_product)
In this function, the binary matrix vector is made.
Every row represents one of all the model words.
A cell is assigned to a one if the model word is also in the model words of the product for each column.

## Function: minhash(n, binary_vector_matrix)   
First, a random seed is taken.
Then two lists are generated from random numbers at a length of n. These lists are assigned to: “a” and “b”.

Then for each row of the binary vector matrix, a list of length n hash values are calculated.
Next, for every column in the binary vector matrix, it's check if this equals to 1.  If so. The hashes calculated earlier are checked if they are lower than the signature matrix. If so, the corresponding cell in the signature matrix is changed to that hash value.

The minimized signature matrix is returned

## Function: create_bands(signature_matrix, r)
This function separates the signature matrix in b bands. The b is depended on the length of the signature matrix divided by r.

For each band, at each column, r rows are combined in a list and appended to the new band.

## Function: LSH(bands)
In this function is being loop through all bands. In search for equal signature vectors in a specific band. If two signature vectors are equal, these are appended to the candidate pairs list.

## Function: estimate(data, candidatepairs, data_c, r,n,t)    
This function applies the Jaccard similarity on the candidate pairs. If the jaccard similarity is higher than the given threshold, then there is said to be found a duplicate pair. A list of all duplicates found is returned. 

## Function: get_realpairs(data)
This function is used to obtain all real pairs, based on the given modelID’s. This list is later compared to the found duplicates, to assess the tp, fn anf fp.

## Function: evaluate(data,candidatepairs, pair_estimates, pair_real)
Lastly, the evaluation function returns all evaluation scores. 
