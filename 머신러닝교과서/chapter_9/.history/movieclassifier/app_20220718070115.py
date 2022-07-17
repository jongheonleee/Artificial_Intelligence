from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# 로컬 디렉토리에서 HashingVectorizer를 임포트
from vectorizer import HashingVectorizer, vect


app = Flask(__name__)

# 분류기 준비
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                                    'pkl_objects',
                                    'classifier.pkl'), 'rb'))

db = os.path.join(cur_dir, 'reviews.sqlite')


def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))

    return label[y], proba


def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    c.execute("INSERT INTO review_db (review, sentiment, data)"
              " VALUES (?, ?, DATETIME('now'))", (document, y))

    conn.commit()
    conn.close()
