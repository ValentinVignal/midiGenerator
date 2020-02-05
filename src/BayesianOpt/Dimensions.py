from skopt.space import Real, Categorical, Integer
import pickle


class Dimensions:
    """

    """
    def __init__(self):
        self.dimensions = []        # Real, Categorical, Integer classes
        self.default_dim = []       # Default values (first values tested)
        self.dimensions_names = []  # The name of the dims

        self.all_dimensions_names = []  # The names of all the parameters (even the one not changeable)
        self.all_default_dim = []       # The default value of all the parameters (even the one not changeable)

    @property
    def nb_dims(self):
        """

        :return: number of changeable parameters
        """
        return len(self.dimensions)

    @property
    def nb_all_dims(self):
        """

        :return: Number of all the parameters
        """
        return len(self.all_dimensions_names)

    @property
    def nb_default(self):
        """

        :return: Number of non changeable parameters
        """
        return self.nb_all_dims - self.nb_dims

    @property
    def default_params_dict(self):
        """

        :return: Return the dictionary of not changeable parameters: d[name] = value
        """
        d = {}
        for i in range(self.nb_all_dims):
            name = self.all_dimensions_names[i]
            if self.is_a_default_param(name):
                d[name] = self.all_default_dim[i]
        return d

    def add_Real(self, mtuple, name, prior='uniform'):
        """
        Add the Real dimension to the dimensions
        But if the parameter is fixed, it only adds a categorical dimension with 1 choice
        :param mtuple:
        :param name:
        :param prior:
        :return:
        """
        self.all_dimensions_names.append(name)
        if len(mtuple) == 1:        # The value cannot change
            self.all_default_dim.append(mtuple[0])
        elif len(mtuple) == 2:      # The value can change
            self.dimensions.append(Real(low=min(mtuple), high=max(mtuple), name=name, prior=prior))
            self.default_dim.append(sum(mtuple) / len(mtuple))
            self.all_default_dim.append(sum(mtuple) / len(mtuple))
            self.dimensions_names.append(name)

    def add_Categorical(self, m_tuple, name):
        """
        Add Categorical dimension to the dimensions
        :param m_tuple:
        :param name:
        :return:
        """
        self.all_dimensions_names.append(name)
        self.all_default_dim.append(m_tuple[0])
        if len(list(m_tuple)) > 1:      # If there is more than one value possible
            self.dimensions.append(Categorical(list(m_tuple), name=name))
            self.default_dim.append(m_tuple[0])
            self.dimensions_names.append(name)

    def get_value_param(self, name, l):
        """

        :param l: list of all the parameters value (changeable and not changeable ones)
        :param name: the name of the dimension
        :return: the value of this parameters in the list
        """
        if self.is_a_default_param(name):
            return self.default_value(name)
        else:
            return l[self.index(name)]

    def is_a_default_param(self, name):
        """

        :param name:
        :return: True if the parameters cannot change
        """
        return name in self.all_dimensions_names and not name in self.dimensions_names

    def index(self, name):
        """

        :param name: The index of the parameters in the list of changeable parameters
        :return:
        """
        return self.dimensions_names.index(name)

    def index_all(self, name):
        """

        :param name:
        :return: The index of the dim in the list of all parameters (changeable and not changeable)
        """
        return self.all_dimensions_names.index(name)

    def default_value(self, name):
        """

        :param name:
        :return: The default value of the dimension
        """
        return self.all_default_dim[self.index_all(name)]

    def point_to_dict(self, x):
        """

        :param x: list of all dimension value
        :return: the value but in a dict with the names of the dimension as a key: d[name] = value
        """
        d = {}
        for i in range(self.nb_dims):
            d[self.dimensions_names[i]] = x[i]
        return d

    def save(self, path):
        """

        :param path:
        :return:
        """
        with open(path, 'wb') as dump_file:
            pickle.dump(
                dict(
                    dimensions=self.dimensions,
                    default_dim=self.default_dim,
                    dimensions_names=self.dimensions_names,
                    all_default_dim=self.all_default_dim,
                    all_dimensions_names=self.all_dimensions_names
                ), dump_file
            )

    def load(self, path):
        """

        :param path:
        :return:
        """
        with open(path, 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.dimensions = d['dimensions']
            self.default_dim = d['default_dim']
            self.dimensions_names = d['dimensions_names']
            self.all_default_dim = d['all_default_dim']
            self.all_dimensions_names = d['all_dimensions_names']


