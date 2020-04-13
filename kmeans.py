"""COMP 527: Implementing the k-means clustering algorithm"""

# import libraries
import numpy as np                 # for handling vectors
import matplotlib.pyplot as plt    # for plotting results
import pandas as pd                # for neatly displaying results in table


# define classes to read in data

class Word():
    """Object class for a categorized word with data vector."""
    
    def __init__(self, name, vector, category):
        """Initialize Word object."""
        self.name = name
        self.vector = vector
        self.category = category
    
    def __repr__(self):
        """Define printable representation."""
        desc = f"{self.name} ({self.category}):" + \
            f"[{self.vector[0]},...,{self.vector[len(self.vector) - 1]}] ({len(self.vector)}-dim.)"
        return desc
    
    def dot(self, other):
        """Allow dot products of word vectors directly."""
        
        return self.vector.dot(other.vector)

    
class Dataset():
    
    def __init__(self, list_of_filenames):
        """Read in Word data from various files."""
        
        self.data = []
        self.categories = list_of_filenames
        self.dim = None
        
        for category in list_of_filenames:
            print(f'Reading in data from {category}.')
            lines = open(category).read().split('\n')[:-1]

            for word_data in lines:
                split = word_data.split(' ')
                name = split[0]
                for other_word in self.data:
                    if other_word.name == name:
                        print(f"'{name}' is already named, but we add it again anyway.\nYou can remove with `.remove('{name}')`")
                
                raw_list = split[1:]

                floats = []
                    
                for x_string in raw_list:
                    floats.append(float(x_string))

                if not self.dim:
                    # define dimensionality of dataset
                    # based on length of first vector
                    self.dim = len(floats)
                else:
                    # require that all vectors are same length
                    assert self.dim == len(floats),\
                    'Data vectors are not all of same length!'

                vector = np.array(floats)

                self.data.append( Word(name, vector, category))
                
    def __repr__(self):
        """Define printable representation of Dataset."""
        
        summary = f'Collection of {len(self.data)} words'\
        + f' from {len(self.categories)} categories,'\
        + f' with {self.dim}-dimensional feature vectors.'
        
        return summary
    
    def select(self, word_name):
        """Select Word from Dataset by name."""
        
        selected = []
        for w in self.data:
            if w.name == word_name:
                selected.append(w)
        
        if len(selected) == 1:
            return selected[0]
        else:
            return selected
    
    def repeats(self):
        """Check for repetitions and return list of tuples."""
        
        repetitions = []
        for i, word1 in enumerate(self.data):
            for word2 in self.data[i:]:
                if word1 != word2:
                    if word1.name == word2.name:
                        repetitions.append((word1, word2))
                        
        return repetitions
                    
    def remove(self, name, word=None):
        """Remove Word, specified by name, or by identical object."""
        
        if word:
            for w in self.data:
                if word == w:
                    self.data.remove(w)
                    print(f'Removed {w}.')
                    return
                
        else:
            for w in self.data:
                if name == w.name:
                    self.data.remove(w)
                    print(f'Removed {w}.')
                    return
        
        print('Nothing was removed.')
        return



def normalize(data):
    """Return normalized vectors (ie. parallel vector with unit magnitude)."""
    
    normalized_data = []
    
    for d in data:
        normalized_vector = d.vector / np.sqrt( d.vector.dot(d.vector) )
        normalized_data.append(Word(d.name, normalized_vector, d.category))
        
    return normalized_data
    

    
# define distance metrics

def euclidean_distance(u,v):
    """Return Euclidean distance between two np.array vectors."""
    
    return np.sqrt( (u - v).dot( u - v ))



def manhattan_distance(u,v):
    """Return Manhattan distance between two np.array vectors."""
    
    w = u - v
    distance = 0
    for x in w:
        distance += abs(x)
    
    return distance



def cosine_similarity(u, v):
    """Return cosine similarity of two np.array vectors."""
    
    if np.array_equal(u, v):
        # we specify this to avoid rounding errors
        cos_theta = 1
    else:
        cos_theta = u.dot(v)/( np.sqrt(u.dot(u)) * np.sqrt(v.dot(v)) )
    
    return cos_theta


def angular_distance(u, v):
    """Return angular distance between two np.array vectors."""

    cos_theta = cosine_similarity(u, v)

    theta = np.arccos(cos_theta)
    
    if theta < 0:
        theta += 2 * np.pi
    
    return theta


# implement K-Means algorithm

class KMeans():
    
    def __init__(
                self, 
                k, 
                D, 
                metric = euclidean_distance, 
                normed = False, 
                max_iterations = 10**3, 
                seed = None,
                ):
        """
        Initialize KMeans Model.
        
        Args:
            k (int): number of clusters to divide data into.
            D (Dataset): as defined by Dataset() class.
            metric (function): to measure distance between points.
            norm (Boolean): whether or not to normalize vectors.
            iterations (int): when to stop if no convergence.
            seed (int): for reproducible (pseudo-)randomness.
        """
        
        self.k = k
        self.D = D
        self.normed = normed
        if self.normed == True:
            self.data = normalize(D.data)
        else:
            self.data = D.data
        
        self.metric = metric
                
        if seed:
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 2**32)
        np.random.seed(seed)
        
        # we track centroid positions and cluster labels in nested dicts,
        # of the form dict_name[iteration_number][centroid_number]
        self._centroid = {}
        
        self._cluster = {}

        # we record cluster labels explicitly as well
        self._label = {}
        
        self.max_iterations = max_iterations
        for i in range(self.max_iterations):
            self._iteration = i
            self._iterate()
            if i > 0 and self._cluster[i] == self._cluster[i-1]:
                break
        
        self.cluster = self._cluster[self._iteration]
        self.label = self._label[self._iteration]
        
        self.convergence = self._iteration + 1
        if self.convergence == max_iterations:
            self.convergence = np.nan
        
        self._evaluate()
        
    def __repr__(self):
        """Representation of model."""
        
        desc = f"k = {self.k}"
    
    
    def _start(self):
        """Shuffle dataset and position centroids on first k datapoints."""
        
        selected = np.random.permutation(self.data)[0:self.k]
        
        self._centroid[0] = {}
        
        for centroid_number in range(self.k):
            self._centroid[0][centroid_number] = selected[centroid_number].vector


    def _classify(self):
        """Assign each data point to cluster of nearest centroid."""
        
        self._cluster[self._iteration] = {}
        self._label[self._iteration] = {}
        
        for centroid_number in range(self.k):
            self._cluster[self._iteration][centroid_number] = []
        
        for d in self.data:
            distances = [] 
            
            for centroid_number in range(self.k):
                
                distances.append(self.metric(d.vector, self._centroid[self._iteration][centroid_number]))
            
            closest_centroid = np.argmin(distances)
            
            self._cluster[self._iteration][closest_centroid].append(d)
            self._label[self._iteration][d.name] = closest_centroid
        
            
    def _reposition(self):
        """Move centroids to mean of each cluster."""
        
        for centroid_number in range(self.k):
            self._centroid[self._iteration] = {}
        
        for centroid_number in range(self.k):
            
            clustered = self._cluster[self._iteration - 1][centroid_number]
            
            if len(clustered) > 0:
                vector_sum = np.zeros(len(clustered[0].vector))
                
                for datum in clustered:
                    vector_sum += datum.vector

                cluster_mean = vector_sum / len(clustered)

                self._centroid[self._iteration][centroid_number] = cluster_mean

            else:
                # if nothing assigned to this cluster, then position is unchanged
                self._centroid[self._iteration][centroid_number] = \
                self._centroid[self._iteration - 1][centroid_number]
    

    def _iterate(self):
        """Position centroids and classify data by nearest centroid."""
        
        if self._iteration == 0:
            self._start()
        else:
            self._reposition()
        
        self._classify()
            
            
    def _evaluate(self):
        """Evaluate success of clustering."""
        
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        data = self.data
        
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                if data[i].category == data[j].category \
                and self.label[data[i].name] == self.label[data[j].name]:
                    self.true_positives += 1
                if data[i].category != data[j].category \
                and self.label[data[i].name] == self.label[data[j].name]:
                    self.false_positives += 1
                if data[i].category != data[j].category \
                and self.label[data[i].name] != self.label[data[j].name]:
                    self.true_negatives += 1
                if data[i].category == data[j].category \
                and self.label[data[i].name] != self.label[data[j].name]:
                    self.false_negatives += 1
                    
        
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.f_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        
        self.score = f'Precision: {self.precision}.\nRecall: {self.recall}.\nF-Score: {self.f_score}.\n'
        
    def describe(self):
        """Describe model clusters."""
        
        desc = ""
        for i in range(self.k):
            desc += f"Cluster {i} includes {len(self.cluster[i])} elements.\n"
            
            count = {}
            for category in self.D.categories:
                count[category] = 0
                for word in self.cluster[i]:
                    if word.category == category:
                        count[category] += 1
                if count[category] > 0:
                       desc += f"{count[category]} are {category}.\n"
            desc += '\n'
            
        print(f'K-Means Model\n\nParameters:\nk={self.k}\nD={self.D}\nmetric={self.metric}\nnormed={self.normed}\nseed={self.seed}\n')
        print(f'Converged after {self.convergence} iterations.\n')
        print(self.score)
        print(desc)


# define functions to avoid repetition while doing requested tasks.

def get_scores(max_k, D, metric, normed=False, seed=1):
    "Get model scores for range of values of k."
    
    scores = {}
    measures = 'precision', 'recall', 'f_score'
    for measure in measures:
        scores[measure] = []
    model = {}
    for k in range(1, max_k+1):
        model[k] = KMeans(k=k, D=D, metric=metric, normed=normed, seed=seed)
        for measure in measures:
            scores[measure].append( getattr(model[k], measure) )
            
    return scores



def show_results(scores):
    """Plot results for given scores and return scores as table."""

    measures = 'precision', 'recall', 'f_score'
    max_k = len(scores[measures[0]])
    
    fig, ax = plt.subplots(figsize=(15,8))

    for measure in measures:
        ax.plot(range(1,max_k + 1), scores[measure], label = measure)

    ax.set_xticks(range(1,max_k + 1))
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Evaluation Score')
    ax.set_yticks(np.arange(0,max_k + 1)/max_k)
    ax.set_xlim(1,max_k)
    ax.set_ylim(0,1.01)
    ax.legend()

    plt.show()

    scores['k'] = list(range(1,10+1))
    table = pd.DataFrame(scores).set_index('k')
    return table.transpose()
    
def compare_metrics(seed):
    """Return results for all metrics (normed and not), given seed."""
    
    metrics = euclidean_distance, manhattan_distance, angular_distance
    
    normed = False, True
    
    results = {}
    
    for m in metrics:
        for boolean in normed:
            if (m == angular_distance) and (boolean == True):
                # don't need to take norm for angular_distance
                pass
            else:
                results[f'{m.__name__}, normed={boolean}'] =\
                KMeans(k=4, D=words, metric=m, normed=boolean, max_iterations=100, seed=seed).f_score
    
    
    
    return results


if __name__ == '__main__':

    words = Dataset(['animals','countries','fruits','veggies'])
    seed = np.random.randint(0,2**32)
    print(f'Seed = {seed}')
    
    print('\nRunning K-means clustering using Euclidean distance...')
    print(show_results(get_scores(
        max_k=10, D=words, metric=euclidean_distance, normed=False, seed=seed)))
    
    print('\nRunning K-means clustering using Euclidean distance, on L2 normed data-vectors...') 
    print(show_results(get_scores(
        max_k=10, D=words, metric=euclidean_distance, normed=True, seed=seed)))
    
    print('\nRunning K-means clustering using Manhattan distance...')
    print(show_results(get_scores(
        max_k=10, D=words, metric=manhattan_distance, normed=False, seed=seed)))

    print('\nRunning K-means clustering using Manhattan distance, on L2 normed data-vectors...') 
    print(show_results(get_scores(
        max_k=10, D=words, metric=manhattan_distance, normed=True, seed=seed)))
    
    print('\nRunning K-means clustering using Angular distance...')
    print(show_results(get_scores(
        max_k=10, D=words, metric=angular_distance, normed=False, seed=seed)))
    
    results = compare_metrics(seed)
    print(f'\nWith a randomising seed of {seed}, our best f-score was given by {max(results, key=lambda key: results[key])}')