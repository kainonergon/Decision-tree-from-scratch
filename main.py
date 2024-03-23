import pandas as pd
from sklearn.metrics import confusion_matrix


class DecisionTree:
    class Node:

        def __init__(self):
            # class initialization
            self.left = None
            self.right = None
            self.term = False
            self.label = None
            self.feature = None
            self.value = None

        def set_split(self, feature, value):
            # this function saves the node splitting feature and its value
            self.feature = feature
            self.value = value

        def set_term(self, label):
            # if the node is a leaf, this function saves its label
            self.term = True
            self.label = label

    def __init__(self, min_samples=1, numerical=None):
        self.root = self.Node()
        self.min_samples = min_samples
        self.target_name = None
        self.target_dtype = None
        self.numerical = [] if numerical is None else numerical

    def _gini_impurity(self, s: pd.Series) -> float:
        if s.empty:
            raise ValueError("Empty series")
        return 1 - (s.value_counts() ** 2).sum() / len(s) ** 2

    def _weighted_gini_impurity(self, s1: pd.Series, s2: pd.Series) -> float:
        if s1.empty or s2.empty:
            raise ValueError("Empty series")
        n1 = len(s1)
        n2 = len(s2)
        return (n1 * self._gini_impurity(s1) + n2 * self._gini_impurity(s2)) / (n1 + n2)

    def _split(self, data: pd.DataFrame, target: pd.Series):
        try_split = None
        for feature in data:
            unique_values = data[feature].unique()
            if len(unique_values) <= 1:
                continue
            for value in unique_values:
                if feature in self.numerical:
                    left = target.loc[data[feature] <= value]
                    right = target.loc[data[feature] > value]
                    if left.empty or right.empty:
                        continue
                else:
                    left = target.loc[data[feature] == value]
                    right = target.loc[data[feature] != value]
                wg = self._weighted_gini_impurity(left, right)
                if try_split is None or wg < try_split[0]:
                    try_split = wg, feature, value, list(left.index), list(right.index)
        if try_split is None:
            raise ValueError("Unable to split")
        return try_split

    def _split_recursive(self, node: Node, data: pd.DataFrame, target: pd.Series) -> None:
        # if it's a leaf -> terminate
        if len(data) <= self.min_samples \
                or self._gini_impurity(target) == 0 \
                or len(data.value_counts()) == 1:
            node.set_term(target.value_counts().index[0])
            return
        # not a leaf -> split
        _, split_feature, split_value, left_index, right_index = self._split(data, target)
        # print(f'Made split: {split_feature} is {split_value}')  # log message
        node.set_split(split_feature, split_value)
        node.left = self.Node()
        self._split_recursive(node.left,
                              data.iloc[left_index].reset_index(drop=True),
                              target[left_index].reset_index(drop=True))
        node.right = self.Node()
        self._split_recursive(node.right,
                              data.iloc[right_index].reset_index(drop=True),
                              target[right_index].reset_index(drop=True))

    def fit(self, data: pd.DataFrame, target: pd.Series) -> None:
        self._split_recursive(self.root, data, target)
        self.target_name = target.name
        self.target_dtype = target.dtype

    def _predict_recursive(self, sample: pd.Series, node: Node):
        if node.term:
            # print(f'   Predicted label: {node.label}')  # log message
            return node.label
        # print(f'   Considering decision rule on feature {node.feature} with value {node.value}')  # log message
        if node.feature in self.numerical:
            next_node = node.left if sample[node.feature] <= node.value else node.right
        else:
            next_node = node.left if sample[node.feature] == node.value else node.right
        return self._predict_recursive(sample, next_node)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.target_name is None:
            raise AttributeError('The model is not fit')
        predicted = pd.Series(name=self.target_name, dtype=self.target_dtype, index=data.index)
        for i in data.index:
            # print(f'Prediction for sample # {i}')  # log message
            predicted[i] = self._predict_recursive(data.iloc[i], self.root)
        return predicted


def get_data(file_name: str, feature_name) -> (pd.DataFrame, pd.Series):
    data = pd.read_csv(file_name, index_col=0)
    return data.drop(columns=feature_name), data[feature_name]


file_train, file_test = input().split()
# file_train, file_test = 'test/data_stage9_train.csv test/data_stage9_test.csv'.split()
target_name = 'Survived'
num_features = ['Fare', 'Age']
leaf_size = 74

X, y = get_data(file_train, target_name)
X_test, y_test = get_data(file_test, target_name)

tree = DecisionTree(leaf_size, num_features)
tree.fit(X, y)
y_predicted = tree.predict(X_test)
conf_matr = confusion_matrix(y_test, y_predicted, normalize='true')

print(conf_matr[1, 1].round(3), conf_matr[0, 0].round(3))
