__author__ = 'tienbm'

from sklearn import cross_validation
from sklearn import svm
from sklearn import datasets


def test_cross_validation():
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    scores = clf.score(X_test, y_test)

if __name__ == '__main__':
    test_cross_validation()
