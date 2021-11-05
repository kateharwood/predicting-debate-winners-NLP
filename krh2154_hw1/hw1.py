import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import features
import scipy


# Time the runtime
from datetime import datetime
startTime = datetime.now()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()

###############################
# READ IN TEST AND TRAIN FILES
###############################
debates_train = pd.read_json(str(args.train), lines=True)
debates_test = pd.read_json(str(args.test), lines=True)


############
# GET LABELS
############
y_train = features.get_debates_labels(debates_train)
y_test = features.get_debates_labels(debates_test)


##########################
# MAJORITY BASELINE MODEL
##########################
# Simple majority baseline model
# def majority_baseline_model_predict(debates):
#     labels = features.get_debates_labels(debates)
#     if labels.count(1) > labels.count(0):
#         return([1] * len(labels))
#     else:
#         return([0] * len(labels))

# # # Predict with baseline model
# y_baseline_model_predicted = majority_baseline_model_predict(debates_test)
# print("SIMPLE MAJORITY BASELINE")
# print(classification_report(y_test, y_baseline_model_predicted))


#############
# NGRAM MODEL
#############
def ngram_model():
    # Get ngram features
    X_ngrams_train, X_ngrams_test = features.get_ngram_features(debates_train, debates_test)

    # Save constructed features
    scipy.sparse.save_npz('X_ngrams_train.npz', X_ngrams_train)
    scipy.sparse.save_npz('X_ngrams_test.npz', X_ngrams_test)

    # Train model
    clf_ngrams = LogisticRegression()
    clf_ngrams.fit(X_ngrams_train, y_train)

    # Predict with test data
    y_ngrams_predicted = clf_ngrams.predict(X_ngrams_test)

    # Evaluate Model
    print("NGRAMS WITH BEST FEATURES SELECTED")
    print(classification_report(y_test, y_ngrams_predicted))

    # Write predictions to output file
    output_file = open(str(args.outfile), "w")
    for prediction in y_ngrams_predicted.tolist():
        output_file.write(prediction + "\n")


###################
# NGRAM + LEX MODEL
###################
def ngram_lex_model():
    with open(str(args.lexicon_path) + 'NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/a-scores.txt') as f:
        nrc_arousal_lexicon = f.readlines()

    # Get features (and save them)
    X_train_ngram_lex, X_test_ngram_lex = features.get_ngrams_lex_features(debates_train, debates_test, nrc_arousal_lexicon)
    scipy.sparse.save_npz('X_train_ngram_lex.npz', X_train_ngram_lex)
    scipy.sparse.save_npz('X_test_ngram_lex.npz', X_test_ngram_lex)

    # Train ngram+lex model
    clf_ngrams_lex = LogisticRegression()
    clf_ngrams_lex.fit(X_train_ngram_lex, y_train)

    # Predict with test data
    y_ngrams_lex_predicted = clf_ngrams_lex.predict(X_test_ngram_lex)

    # Evaluate Model
    print("NGRAMS PLUS LEX")
    print(classification_report(y_test, y_ngrams_lex_predicted))


    # Write predictions to output file
    output_file = open(str(args.outfile), "w")
    for prediction in y_ngrams_lex_predicted.tolist():
        output_file.write(prediction + "\n")




##########################
# NGRAM + LEX + LING MODEL
##########################
def ngram_lex_ling_model():
    with open(str(args.lexicon_path) + 'NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/a-scores.txt') as f:
        nrc_arousal_lexicon = f.readlines()

    # Get features (and save them)
    X_train_ngram_lex_ling, X_test_ngram_lex_ling, X_train_ngram_lex_ling_ling2, X_test_ngram_lex_ling_ling2 = features.get_ngrams_lex_ling_features(debates_train, debates_test, nrc_arousal_lexicon)
    scipy.sparse.save_npz('X_train_ngram_lex_ling.npz', X_train_ngram_lex_ling)
    scipy.sparse.save_npz('X_test_ngram_lex_ling.npz', X_test_ngram_lex_ling)
    scipy.sparse.save_npz('X_train_ngram_lex_ling_ling2.npz', X_train_ngram_lex_ling_ling2)
    scipy.sparse.save_npz('X_test_ngram_lex_ling_ling2.npz', X_test_ngram_lex_ling_ling2)

    # Train ngram+lex+ling model
    clf_ngrams_lex_ling = LogisticRegression()
    clf_ngrams_lex_ling.fit(X_train_ngram_lex_ling, y_train)

    # Predict with test data
    y_ngrams_lex_ling_predicted = clf_ngrams_lex_ling.predict(X_test_ngram_lex_ling)

    # Evaluate Model
    print("NGRAMS PLUS LEX PLUS LING")
    print(classification_report(y_test, y_ngrams_lex_ling_predicted))

    # Train ngram+lex+ling+ling2 model
    clf_ngrams_lex_ling_ling2 = LogisticRegression(max_iter=200)
    clf_ngrams_lex_ling_ling2.fit(X_train_ngram_lex_ling_ling2, y_train)

    # Predict with test data
    y_ngrams_lex_ling_ling2_predicted = clf_ngrams_lex_ling_ling2.predict(X_test_ngram_lex_ling_ling2)
    
    # Evaluate Model
    print("NGRAMS PLUS LEX PLUS LING, LING2")
    print(classification_report(y_test, y_ngrams_lex_ling_ling2_predicted))

    ##################################################################
    # Ngram + lex + ling + ling2 religious non-religious split model
    #################################################################
    # Predict each test example one by one
    # For non-religious debates: 
    #   use the classifier with ngrams + lex + length linguistic features
    # For religious debates:
    #   use the classifier with ngrams + lex + the length and the PRP frequency linguistic features
    y_ngrams_lex_ling_ling2_religious_split_predicted = []
    for i, test_category in enumerate(debates_test['category']):
        if test_category == 'Religion':
            y_ngrams_lex_ling_ling2_religious_split_predicted.append(clf_ngrams_lex_ling_ling2.predict(X_test_ngram_lex_ling_ling2.getrow(i))[0])
        else:
            y_ngrams_lex_ling_ling2_religious_split_predicted.append(clf_ngrams_lex_ling.predict(X_test_ngram_lex_ling.getrow(i))[0])

    # Evaluate Model
    print("NGRAMS PLUS LEX PLUS LING, LING2 RELIGIOUS SPLIT")
    print(classification_report(y_test, y_ngrams_lex_ling_ling2_religious_split_predicted))

    # Write predictions to output file
    output_file = open(str(args.outfile), "w")
    for prediction in y_ngrams_lex_ling_ling2_religious_split_predicted:
        output_file.write(prediction + "\n")



def ngram_lex_ling_user_model():
    with open(str(args.lexicon_path) + 'NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/a-scores.txt') as f:
        nrc_arousal_lexicon = f.readlines()

    df_user_data = pd.read_json(str(args.user_data), orient="index")

    # These are multiple feature options from combinations of the 2 linguistic and 2 user features
    X_train_ngram_lex_ling_user, X_test_ngram_lex_ling_user, X_train_ngram_lex_ling_ling2_user, X_test_ngram_lex_ling_ling2_user, X_train_ngram_lex_ling_user_user2, X_test_ngram_lex_ling_user_user2, X_train_ngram_lex_ling_ling2_user_user2, X_test_ngram_lex_ling_ling2_user_user2 = features.get_ngrams_lex_ling_user_features(debates_train, debates_test, df_user_data, nrc_arousal_lexicon)

    # Train ngram+lex+ling+user model
    clf_ngrams_lex_ling_user = LogisticRegression(solver='liblinear')
    clf_ngrams_lex_ling_user.fit(X_train_ngram_lex_ling_user, y_train)

    # Train ngram+lex+ling+ling2+user model
    clf_ngrams_lex_ling_ling2_user = LogisticRegression(solver='liblinear')
    clf_ngrams_lex_ling_ling2_user.fit(X_train_ngram_lex_ling_ling2_user, y_train)

    # Train ngram+lex+ling+user+user2 model
    clf_ngrams_lex_ling_user_user2 = LogisticRegression(solver='liblinear')
    clf_ngrams_lex_ling_user_user2.fit(X_train_ngram_lex_ling_user_user2, y_train)

    # These are the best features and the best model 
    # Train ngram+lex+ling+ling2+user+user2 model
    clf_ngrams_lex_ling_ling2_user_user2 = LogisticRegression(solver='liblinear')
    clf_ngrams_lex_ling_ling2_user_user2.fit(X_train_ngram_lex_ling_ling2_user_user2, y_train)

    clf_ngrams_lex_ling_ling2 = LogisticRegression(solver='liblinear')
    clf_ngrams_lex_ling_ling2.fit(X_train_ngram_lex_ling_ling2_user_user2, y_train)


    # Predict with test data
    y_ngrams_lex_ling_user_predicted = clf_ngrams_lex_ling_user.predict(X_test_ngram_lex_ling_user)
    y_ngrams_lex_ling_ling2_user_predicted = clf_ngrams_lex_ling_ling2_user.predict(X_test_ngram_lex_ling_ling2_user)

    y_ngrams_lex_ling_user_user2_predicted = clf_ngrams_lex_ling_user_user2.predict(X_test_ngram_lex_ling_user_user2)
    
    # These are the best predictions
    y_ngrams_lex_ling_ling2_user_user2_predicted = clf_ngrams_lex_ling_ling2_user_user2.predict(X_test_ngram_lex_ling_ling2_user_user2)


    # Evaluate Models
    print("NGRAMS PLUS LEX PLUS LING PLUS USER")
    print(classification_report(y_test, y_ngrams_lex_ling_user_predicted))
    print("NGRAMS PLUS LEX PLUS LING PLUS LING2 PLUS USER")
    print(classification_report(y_test, y_ngrams_lex_ling_ling2_user_predicted))

    print("NGRAMS PLUS LEX PLUS LING PLUS USER PLUS USER2")
    print(classification_report(y_test, y_ngrams_lex_ling_user_user2_predicted))
    print("NGRAMS PLUS LEX PLUS LING PLUS LING2 PLUS USER PLUS USER2")
    print(classification_report(y_test, y_ngrams_lex_ling_ling2_user_user2_predicted))


    # ########################################################################
    # # Ngram + lex + ling + ling2 + user religious non-religious split model
    # ########################################################################
    # Predict each test example one by one
    # For non-religious debates: 
    #   use the classifier with ngrams + lex + length linguistic features (+ user features)
    # For religious debates:
    #   use the classifier with ngrams + lex + the length and the PRP frequency linguistic features (+ user features)
    y_ngrams_lex_ling_ling2_user_religious_split_predicted = []
    for i, test_category in enumerate(debates_test['category']):
        if test_category == 'Religion':
            y_ngrams_lex_ling_ling2_user_religious_split_predicted.append(clf_ngrams_lex_ling_ling2_user_user2.predict(X_test_ngram_lex_ling_ling2_user_user2.getrow(i))[0])
        else:
            y_ngrams_lex_ling_ling2_user_religious_split_predicted.append(clf_ngrams_lex_ling_user_user2.predict(X_test_ngram_lex_ling_user_user2.getrow(i))[0])

    # Evaluate Model
    print("NGRAMS PLUS LEX PLUS LING, LING2 RELIGIOUS SPLIT PLUS USER,USER2")
    print(classification_report(y_test, y_ngrams_lex_ling_ling2_user_user2_predicted))

    # # Write predictions to output file
    output_file = open(str(args.outfile), "w")
    for prediction in y_ngrams_lex_ling_ling2_user_religious_split_predicted:
        output_file.write(prediction + "\n")


if (args.model == 'Ngram'):
    ngram_model()
if (args.model == 'Ngram+Lex'):
    ngram_lex_model()
if (args.model == 'Ngram+Lex+Ling'):
    ngram_lex_ling_model()
if (args.model == 'Ngram+Lex+Ling+User'):
    ngram_lex_ling_user_model()

# Output the runtime
print(datetime.now() - startTime)
