# COMP527 Data Mining: K-Means Clustering.

In this folder you shuld find:
- this README.md text file,
- a PDF providing the answers to the assignment tasks,
- a Python script called `kmeans.py`,
- and I have also included the data files: `animals`, `countries`, `fruits`, and `veggies`.

To run the program you need to have NumPy (for handling vectors), Matplotlib (for plotting graphs), and Pandas (for neatly storing data in tables) installed in your Python environment:
```
pip install numpy matplotlib pandas
```

If you run the file directly from the command line, `python kmeans.py`, then the program will first read in the data, then explicitly set a seed for reproducible pseudo-randomness, and then run the K-Means clustering algorithm with Euclidean distance, Manhattan distance, and Angular distance (which we use instead of Cosine Similarity, since Cosine Similarity does not give zero distance between a point and itself, and is therefore not a suitable metric to be directly used by our implementation of the algorithm). In each case the results will be plotted by matploblib, and then returned in tabular form. Finally, the version that gives the best f-score will be identified.

```
$ python kmeans.py 
Reading in data from animals.
Reading in data from countries.
Reading in data from fruits.
Reading in data from veggies.
'cucumber' is already named, but we add it again anyway.
You can remove with `.remove('cucumber')`
Seed = 196683050

Running K-means clustering using Euclidean distance...
k                1         2         3         4         5         6         7         8         9         10
precision  0.324857  0.651405  0.834238  0.844901  0.775174  0.906233  0.885691  0.927656  0.925445  0.968838
recall     1.000000  1.000000  0.990586  0.920242  0.590712  0.575650  0.437186  0.421383  0.382417  0.372490
f_score    0.490403  0.788910  0.905715  0.880963  0.670487  0.704068  0.585409  0.579521  0.541197  0.538097

Running K-means clustering using Euclidean distance, on L2 normed data-vectors...
k                1         2         3         4         5         6         7         8         9         10
precision  0.324857  0.651405  0.834296  0.832332  0.763201  0.865670  0.838539  0.863420  0.966870  0.971599
recall     1.000000  1.000000  0.996748  0.932907  0.615986  0.571714  0.434961  0.414765  0.397935  0.380591
f_score    0.490403  0.788910  0.908316  0.879755  0.681736  0.688634  0.572802  0.560351  0.563819  0.546938

Running K-means clustering using Manhattan distance...
k                1         2         3         4         5         6         7         8         9         10
precision  0.324857  0.651405  0.833633  0.946009  0.933099  0.923530  0.897500  0.921581  0.954041  0.972132
recall     1.000000  1.000000  0.978834  0.942663  0.620664  0.590484  0.428115  0.408318  0.392001  0.358227
f_score    0.490403  0.788910  0.900417  0.944333  0.745469  0.720376  0.579706  0.565905  0.555681  0.523534

Running K-means clustering using Manhattan distance, on L2 normed data-vectors...
k                1         2         3         4         5         6         7         8         9         10
precision  0.324857  0.651405  0.833633  0.969462  0.952701  0.951725  0.938177  0.969482  0.964208  0.970040
recall     1.000000  1.000000  0.978834  0.968964  0.614788  0.583752  0.449338  0.413225  0.379621  0.362049
f_score    0.490403  0.788910  0.900417  0.969213  0.747321  0.723647  0.607646  0.579463  0.544762  0.527295

Running K-means clustering using Angular distance...
k                1         2         3         4         5         6         7         8         9         10
precision  0.324857  0.651405  0.834238  0.835289  0.763200  0.961500  0.951774  0.950345  0.969701  0.971544
recall     1.000000  1.000000  0.990586  0.942036  0.598699  0.604119  0.477408  0.440039  0.399874  0.383729
f_score    0.490403  0.788910  0.905715  0.885457  0.671015  0.742020  0.635866  0.601544  0.566247  0.550162

With a randomising seed of 196683050, our best f-score was given by manhattan_distance, normed=True
```

Or you can import the program from the Python interpreter and play with it. 

First, pass in a list of data files to `Dataset()` to load some data.
```
$ python

>>> from kmeans import *
>>> fruit_and_veg = Dataset(['fruits','veggies'])
Reading in data from fruits.
Reading in data from veggies.
'cucumber' is already named, but we add it again anyway.
You can remove with `.remove('cucumber')`
>>> fruit_and_veg
Collection of 118 words from 2 categories, with 300-dimensional feature vectors.
>>> fruit_and_veg.data[:3]
[apple (fruits):[-0.077424,...,0.030694] (300-dim.), apricot (fruits):[-0.36757,...,0.054696] (300-dim.), avocado (fruits):[-0.29165,...,-0.34656] (300-dim.)]
```

When the data is read in, we are informed of duplicates. We can remind ourselves of these again by using `.repeats()`, which returns a list of 2-tuples. Or we can use `.select()`, to get any word(s) with a specified name. We can then, if we want, delete duplicates, or indeed any words we like.
```
>>> fruit_and_veg.select('cucumber')
[cucumber (fruits):[-0.88559,...,0.22113] (300-dim.), cucumber (veggies):[-0.88559,...,0.22113] (300-dim.)]
>>> fruit_and_veg.repeats()
[(cucumber (fruits):[-0.88559,...,0.22113] (300-dim.), cucumber (veggies):[-0.88559,...,0.22113] (300-dim.))]
>>> fruit_and_veg.remove('cucumber')
Removed cucumber (fruits):[-0.88559,...,0.22113] (300-dim.).
>>> fruit_and_veg.select('cucumber')
cucumber (veggies):[-0.88559,...,0.22113] (300-dim.)
>>> fruit_and_veg.remove('cucumber')
Removed cucumber (veggies):[-0.88559,...,0.22113] (300-dim.).
>>> fruit_and_veg.select('cucumber')
[]
```

We can select Words from the dataset either by specifying their `name`, or by their index in the list of data.
```
>>> avocado = fruit_and_veg.select('avocado')
>>> avocado
avocado (fruits):[-0.29165,...,-0.34656] (300-dim.)
>>> first_fruit = fruit_and_veg.data[0]
>>> first_fruit
apple (fruits):[-0.077424,...,0.030694] (300-dim.)
```

We can then evaluate the distance between Words' vectors, using any of the defined distance metrics.
```
>>> euclidean_distance(first_fruit.vector, avocado.vector)
8.926843978072794
>>> manhattan_distance(first_fruit.vector, avocado.vector)
120.23133025200005
>>> angular_distance(first_fruit.vector, avocado.vector)
1.2413256837575777
```

But what we really want to do is do some K-Means clustering:
```
>>> clustering = KMeans(k=2, D=fruit_and_veg, metric=euclidean_distance)
>>> clustering
<kmeans.KMeans object at 0x7f1bf39b9c50>
>>>
>>> print(clustering.score)
Precision: 0.7658753709198813.
Recall: 0.7804656788630179.
F-Score: 0.773101692376816.
>>>
>>> clustering.describe()
K-Means Model

Parameters:
k=2
D=Collection of 116 words from 2 categories, with 300-dimensional feature vectors.
metric=<function euclidean_distance at 0x7f1bf55e2cb0>
normed=False
seed=2786407911

Converged after 10 iterations.

Precision: 0.7658753709198813.
Recall: 0.7804656788630179.
F-Score: 0.773101692376816.

Cluster 0 includes 50 elements.
3 are fruits.
47 are veggies.

Cluster 1 includes 66 elements.
54 are fruits.
12 are veggies.
```

If we want, we can look into the specifics of the clusters. And we see that `avocado`, `lemon`, and `olive`, are the most `veggie`-like of the `fruits`.
```
>>> clustering.cluster[0]
[avocado (fruits):[-0.29165,...,-0.34656] (300-dim.), lemon (fruits):[-0.45297,...,0.037585] (300-dim.), olive (fruits):[0.07111,...,-0.058863] (300-dim.), aubergine (veggies):[-0.31617,...,-0.77578] (300-dim.), asparagus (veggies):[-0.4557,...,0.20877] (300-dim.), legumes (veggies):[0.65255,...,0.33024] (300-dim.), beans (veggies):[0.24658,...,0.32034] (300-dim.), chickpeas (veggies):[0.37457,...,0.0074993] (300-dim.), lentils (veggies):[0.42484,...,0.19859] (300-dim.), peas (veggies):[0.065506,...,0.98643] (300-dim.), broccoli (veggies):[-0.16262,...,0.2959] (300-dim.), cabbage (veggies):[-0.10551,...,0.21283] (300-dim.), kohlrabi (veggies):[-0.48157,...,0.42538] (300-dim.), cauliflower (veggies):[-0.14071,...,-0.018724] (300-dim.), celery (veggies):[-0.22796,...,-0.013369] (300-dim.), endive (veggies):[-0.10372,...,-0.579] (300-dim.), frisee (veggies):[0.56364,...,-0.89242] (300-dim.), fennel (veggies):[-0.15919,...,0.25369] (300-dim.), greens (veggies):[-0.61894,...,0.11367] (300-dim.), kale (veggies):[-0.044237,...,-0.023015] (300-dim.), spinach (veggies):[-0.14077,...,-0.088264] (300-dim.), basil (veggies):[0.0058528,...,0.23033] (300-dim.), caraway (veggies):[-0.013987,...,-0.36792] (300-dim.), cilantro (veggies):[-0.066871,...,0.35847] (300-dim.), coriander (veggies):[0.23117,...,0.034191] (300-dim.), dill (veggies):[-0.12973,...,0.37783] (300-dim.), marjoram (veggies):[0.1764,...,0.61674] (300-dim.), oregano (veggies):[0.13827,...,0.25633] (300-dim.), parsley (veggies):[-0.34477,...,0.40299] (300-dim.), rosemary (veggies):[0.072254,...,0.29457] (300-dim.), sage (veggies):[-0.42948,...,0.27685] (300-dim.), thyme (veggies):[0.35429,...,0.66731] (300-dim.), lettuce (veggies):[0.10834,...,-0.0041712] (300-dim.), arugula (veggies):[0.090111,...,-0.28214] (300-dim.), mushrooms (veggies):[-0.030242,...,0.63676] (300-dim.), okra (veggies):[-0.24064,...,0.21243] (300-dim.), onions (veggies):[-0.39603,...,0.42284] (300-dim.), chives (veggies):[-0.25527,...,0.61708] (300-dim.), garlic (veggies):[-0.32817,...,0.47746] (300-dim.), leek (veggies):[0.0040342,...,0.14252] (300-dim.), onion (veggies):[-0.46813,...,0.59411] (300-dim.), shallot (veggies):[0.15236,...,0.68791] (300-dim.), peppers (veggies):[-0.20205,...,0.46459] (300-dim.), paprika (veggies):[-0.19896,...,-0.15895] (300-dim.), radicchio (veggies):[0.011026,...,-0.6474] (300-dim.), turnip (veggies):[-0.52,...,0.23786] (300-dim.), radish (veggies):[-0.35831,...,0.22921] (300-dim.), courgette (veggies):[-0.25113,...,-0.27661] (300-dim.), potato (veggies):[-0.43023,...,-0.35875] (300-dim.), zucchini (veggies):[-0.58577,...,0.087304] (300-dim.)]
```