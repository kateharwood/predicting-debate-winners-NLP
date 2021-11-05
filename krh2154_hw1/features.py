
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import nltk
import os
# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
nltk.download('averaged_perceptron_tagger')


######################################################################
# READ ME
# HOW TO EXTRACT THE FEATURES
######################################################################
# There are 5 main functions used for feature extraction in this file:
#
# 1. get_debate_labels
#   Input: All debates from the given data
#   Output: A list of the winner labels for all the debates
#
# 2. get_ngram_features
#   Input: 
#       1. A list of training debates
#       2. A list of testing debates
#   Output: 2 feature objects
#       Vectorized ngram features for training and testing data
#	        Where ngrams are the top 200 1,2,3ngrams and their values are their Tdidf score
#
# 3. get_ngram_lex_features
#   Input: 
#       1. A list of training debates
#       2. A list of testing debates
#       3. The NRC-VAD-Lexicon file
#   Output: 2 feature objects
#       The vectorized ngrams+lex features for training and testing data
#	        Where lex is the frequency of high arousal words in the pro and con sides
#
# 4. get_ngram_lex_ling_features
#   Input: 
#       1. A list of training debates
#       2. A list of testing debates
#       3. The NRC-VAD-Lexicon file
#   Output: 4 feature objects
#       The vectorized ngrams+lex+ling1 features for training and testing data
#       And the vectorized ngrams+lex+ling2 features for training and testing data
#           Where ling1 is length
#           And ling2 is personal pronouns
#
# 5. get_ngram_lex_ling_user_features
#   Input: 
#       1. A list of training debates
#       2. A list of testing debates
#       3. The user data in pd df
#       3. The NRC-VAD-Lexicon file
#   Output: 8 feature objects
#       The vectorized ngrams+lex+ling1+user features for training and testing data
#       The vectorized ngrams+lex+ling1+ling2+user1 features for training and testing data
#       The vectorized ngrams+lex+ling1+user+user2 features for training and testing data
#       The vectorized ngrams+lex+ling1+ling2+user1+user2 features for training and testing data
#           Where user1 is big issues similarity between voters and debaters
#           And user2 is religious similarity between voters and debaters



######################
# 1: GET DEBATE LABELS
######################
# Get winner labels for all debates
def get_debates_labels(debates):
    winner_labels = []
    for winner in debates['winner']:
        winner_labels.append(winner)
    return(winner_labels)


#######################
# PREPARE DEBATE TEXT
#######################
# Parse debates into list of just the text in each debate 
def get_debates_text(debates):
    all_debates_text = []
    for debate in debates['rounds']:
        debate_text = ""
        for round in debate:
            for side in round:
                debate_text = debate_text + side['text']
                
        all_debates_text.append(debate_text)
    return(all_debates_text)



###############################
# 2: NGRAM FEATURE PREPARATION
###############################
def get_ngram_features(debates_train, debates_test):
    debates_train_text = [debate.lower() for debate in get_debates_text(debates_train)] 
    debates_test_text = [debate.lower() for debate in get_debates_text(debates_test)] 

    # Vectorize input
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.9, stop_words='english', max_features=200, strip_accents='ascii', token_pattern=r'(?u)\b[A-Za-z]+\b')
    X_ngrams_train = vectorizer.fit_transform(debates_train_text)
    X_ngrams_test = vectorizer.transform(debates_test_text)
    return(X_ngrams_train, X_ngrams_test)


################################
# 3: LEXICON FEATURE PREPARATION
################################
def get_ngrams_lex_features(debates_train, debates_test, nrc_arousal_lexicon):

    debates_train_text = [debate.lower() for debate in get_debates_text(debates_train)] 
    debates_test_text = [debate.lower() for debate in get_debates_text(debates_test)] 

    # Extract the word score pairs from an NRC-VAD text file
    # that have a score greater than the given min_score param
    # Takes in int and an nrc_vad_lexicon text file with word score pairs
    def get_high_vad_words(min_score, lexicon):
        word_score_pairs = []
        for word in lexicon:
            word_score_pair = word.removesuffix('\n').split('\t')
            word_score_pair[1] = float(word_score_pair[1])
            if word_score_pair[1] > min_score:
                word_score_pairs.append(word_score_pair)
        return(word_score_pairs)


    # Get the words that appear in the debates that are of high arousal
    # (Score greater than min_score)
    def get_vad_words(min_score, lexicon):
        high_vad_words = get_high_vad_words(min_score, lexicon)
        tokenized_debates = [nltk.word_tokenize(debate) for debate in debates_train_text]
        vad_features = []
        for word in high_vad_words:
            for debate in tokenized_debates:
                if word[0] in debate:
                    vad_features.append(word[0])
                    break
        return(list(set(vad_features)))

    # Create matrices of debates as rows and high vad words as columns
    # with the values being the frequency of those words in each debate
    arousal_words = get_vad_words(.97, nrc_arousal_lexicon)
    vad_words = arousal_words 
    vad_vectorizer = CountVectorizer(vocabulary=vad_words)
    X_train_vad = vad_vectorizer.fit_transform(debates_train_text)
    X_test_vad = vad_vectorizer.transform(debates_test_text)

    # Check if previous feature already exists and load it, otherwise create it
    if os.path.exists('X_ngrams_train.npz') and os.path.exists('X_ngrams_test.npz'):
        X_ngrams_train = scipy.sparse.load_npz('X_ngrams_train.npz')
        X_ngrams_test = scipy.sparse.load_npz('X_ngrams_test.npz')
    else:
        X_ngrams_train, X_ngrams_test = get_ngram_features(debates_train, debates_test)
    
    # Concat arousal feature matrix to the original ngram matrix
    X_train_ngram_lex = scipy.sparse.hstack([X_ngrams_train, X_train_vad])
    X_test_ngram_lex = scipy.sparse.hstack([X_ngrams_test, X_test_vad])

    return(X_train_ngram_lex, X_test_ngram_lex)




############################################
# 4: NGRAM + LEX + LING FEATURE PREPARATION
############################################
def get_ngrams_lex_ling_features(debates_train, debates_test, nrc_arousal_lexicon):
    # LING FEATURE 1: LENGTH
    # Get total length of pro and con sides for each debate
    def get_length_of_sides(debates):
        all_pro_lengths = []
        all_con_lengths = []
        for debate in debates['rounds']:
            pro_length = 0
            con_length = 0
            for round in debate:
                for side in round:
                    if side['side'] == "Pro":
                        pro_length = pro_length + len(side['text'])
                    if side['side'] == "Con":
                        con_length = con_length + len(side['text'])
            all_pro_lengths.append(pro_length)
            all_con_lengths.append(con_length)
        return(np.array(all_pro_lengths), np.array(all_con_lengths))

    # Hstack lengths onto the ngram + lex feature matrix
    lengths_of_debates_train = np.transpose(get_length_of_sides(debates_train))
    lengths_of_debates_train = scipy.sparse.csr_matrix(lengths_of_debates_train)

    lengths_of_debates_test = np.transpose(get_length_of_sides(debates_test))
    lengths_of_debates_test = scipy.sparse.csr_matrix(lengths_of_debates_test)

    # Check if previous features already exist and load them, otherwise create them
    if os.path.exists('X_train_ngram_lex.npz') and os.path.exists('X_test_ngram_lex.npz'):
        X_train_ngram_lex = scipy.sparse.load_npz('X_train_ngram_lex.npz')
        X_test_ngram_lex = scipy.sparse.load_npz('X_test_ngram_lex.npz')
    else:
        X_train_ngram_lex, X_test_ngram_lex = get_ngrams_lex_features(debates_train, debates_test, nrc_arousal_lexicon)

    X_train_ngram_lex_ling = scipy.sparse.hstack([X_train_ngram_lex, lengths_of_debates_train])
    X_test_ngram_lex_ling = scipy.sparse.hstack([X_test_ngram_lex, lengths_of_debates_test])


    # LING FEATURE 2: PRP
    # Extract only the personal pronouns from the part of speech tokens
    def get_prp_counts(tokenized_text):
        pos_tokens = nltk.pos_tag_sents(tokenized_text)
        prp_counts = []
        for debate in pos_tokens:
            prp_count = 0
            for token in debate:
                if token[1] == 'PRP':
                    prp_count = prp_count + 1
            prp_counts.append(prp_count)
        return(np.array(prp_counts))

    debates_train_text = [debate.lower() for debate in get_debates_text(debates_train)] 
    debates_test_text = [debate.lower() for debate in get_debates_text(debates_test)] 

    tokenized_text_train = [nltk.word_tokenize(debate) for debate in debates_train_text]
    tokenized_text_test = [nltk.word_tokenize(debate) for debate in debates_test_text]

    prp_counts_train = get_prp_counts(tokenized_text_train)
    prp_counts_test = get_prp_counts(tokenized_text_test)


    # Hstack prp counts onto the ngram + lex + ling1 feature matrix
    prp_counts_train_reshaped = prp_counts_train.reshape(prp_counts_train.shape+(1,))
    prp_counts_train_reshaped = scipy.sparse.csr_matrix(prp_counts_train_reshaped)

    prp_counts_test_reshaped = prp_counts_test.reshape(prp_counts_test.shape+(1,))
    prp_counts_test_reshaped = scipy.sparse.csr_matrix(prp_counts_test_reshaped)

    X_train_ngram_lex_ling_ling2 = scipy.sparse.hstack([X_train_ngram_lex_ling, prp_counts_train_reshaped])
    X_test_ngram_lex_ling_ling2 = scipy.sparse.hstack([X_test_ngram_lex_ling, prp_counts_test_reshaped])

    # Return the linguistics features in separate training matrices
    return(X_train_ngram_lex_ling, X_test_ngram_lex_ling, X_train_ngram_lex_ling_ling2, X_test_ngram_lex_ling_ling2)



###########################################################
# 5: NGRAM + LEX + LING + LING2 + USER FEATURE PREPARATION
###########################################################
def get_ngrams_lex_ling_user_features(debates_train, debates_test, df_user_data, nrc_arousal_lexicon):

    # USER FEATURES 1
    # Gets the voters and pro and con debater "scores" for a given "big issue"
    # where the score is 1 if the user is for the given issue, and -1 if the user is against the issue.
    # (The voters views are aggregated into one score) 
    def get_user_scores(debates, df_user_data, big_issue):
        df_user_data_subset = df_user_data["big_issues_dict"]
        pro_issue_users = df_user_data_subset[[df_user_data_subset[i][big_issue] == "Pro" for i in range(len(df_user_data_subset))]]
        con_issue_users = df_user_data_subset[[df_user_data_subset[i][big_issue] == "Con" for i in range(len(df_user_data_subset))]]
        voters_big_issue_scores = []
        for voters in debates['voters']:
            voters_score = 0
            for voter in voters:
                if voter in pro_issue_users:
                    voters_score = voters_score + 1
                elif voter in con_issue_users:
                    voters_score = voters_score - 1
            voters_big_issue_scores.append(voters_score)
        pro_debaters_scores = []
        con_debaters_scores = []
        for pro_debater in debates['pro_debater']:
            if pro_debater in pro_issue_users:
                pro_debaters_scores.append(1)
            elif pro_debater in con_issue_users:
                pro_debaters_scores.append(-1)
            else:
                pro_debaters_scores.append(0)
        for con_debater in debates['con_debater']:
            if con_debater in pro_issue_users:
                con_debaters_scores.append(1)
            elif pro_debater in con_issue_users:
                con_debaters_scores.append(-1)
            else:
                con_debaters_scores.append(0)
        return(np.array(voters_big_issue_scores), np.array(pro_debaters_scores), np.array(con_debaters_scores))

    # Get the sums of the voters and debaters scores on all the "big issues"
    # Returns 3 new features: summed voter scores, summed pro debater scores, summed con debaters scores
    def get_users_scores_all_issues(debates):
        aggregated_users_scores = np.zeros(shape=(3,len(debates)))
        for issue in df_user_data["big_issues_dict"][0].keys():
            aggregated_users_scores = np.add(aggregated_users_scores, get_user_scores(debates, df_user_data, issue))
        user_scores = scipy.sparse.csr_matrix(np.transpose(aggregated_users_scores))
        return(user_scores)

    # USER FEATURES 2
    # Get the number of voters in each debate that have the same religion as
    # the pro debater and the number that have the same religion as the con debater
    def get_voters_religious_scores(debates):
        df_user_data_subset = df_user_data["religious_ideology"]
        pro_debaters_religions = []
        for pro_debater in debates['pro_debater']:
            if pro_debater in df_user_data_subset.keys():
                pro_debaters_religions.append(df_user_data_subset[pro_debater])
            else:
                pro_debaters_religions.append(" ")
        con_debaters_religions = []
        for con_debater in debates['con_debater']:
            if con_debater in df_user_data_subset.keys():
                con_debaters_religions.append(df_user_data_subset[con_debater])
            else:
                con_debaters_religions.append(" ")
        
        voter_same_religion_counts_as_pro = []
        voter_same_religion_counts_as_con = []
        for i, voters in enumerate(debates['voters']):
            voter_same_religion_count_pro = 0
            voter_same_religion_count_con = 0
            for voter in voters:
                if voter in df_user_data_subset.keys():
                    if df_user_data_subset[voter] == pro_debaters_religions[i]:
                        voter_same_religion_count_pro = voter_same_religion_count_pro + 1
                    elif df_user_data_subset[voter] == con_debaters_religions[i]:
                        voter_same_religion_count_con = voter_same_religion_count_con + 1
            voter_same_religion_counts_as_pro.append(voter_same_religion_count_pro)
            voter_same_religion_counts_as_con.append(voter_same_religion_count_con)
        voter_religion_counts = np.array(voter_same_religion_counts_as_pro), np.array(voter_same_religion_counts_as_con)
        return(scipy.sparse.csr_matrix(np.transpose(voter_religion_counts)))
    
    # Get new user features
    users_scores_all_isues_train = get_users_scores_all_issues(debates_train)
    users_scores_all_isues_test = get_users_scores_all_issues(debates_test)

    voters_religious_scores_train = get_voters_religious_scores(debates_train)
    voters_religious_scores_test = get_voters_religious_scores(debates_test)

    # Check if previous features already exist and load them, otherwise create them
    if os.path.exists('X_train_ngram_lex_ling.npz') and os.path.exists('X_test_ngram_lex_ling.npz') and os.path.exists('X_train_ngram_lex_ling_ling2.npz') and os.path.exists('X_test_ngram_lex_ling_ling2.npz'):
        X_train_ngram_lex_ling = scipy.sparse.load_npz('X_train_ngram_lex_ling.npz')
        X_test_ngram_lex_ling = scipy.sparse.load_npz('X_test_ngram_lex_ling.npz')
        X_train_ngram_lex_ling_ling2 = scipy.sparse.load_npz('X_train_ngram_lex_ling_ling2.npz')
        X_test_ngram_lex_ling_ling2 = scipy.sparse.load_npz('X_test_ngram_lex_ling_ling2.npz')

    else:
        X_train_ngram_lex_ling, X_test_ngram_lex_ling, X_train_ngram_lex_ling_ling2, X_test_ngram_lex_ling_ling2 = get_ngrams_lex_ling_features(debates_train, debates_test, nrc_arousal_lexicon)

    # Hstack the user features to the already existing features
    X_train_ngram_lex_ling_user = scipy.sparse.hstack([X_train_ngram_lex_ling, users_scores_all_isues_train])
    X_test_ngram_lex_ling_user = scipy.sparse.hstack([X_test_ngram_lex_ling, users_scores_all_isues_test])

    X_train_ngram_lex_ling_ling2_user = scipy.sparse.hstack([X_train_ngram_lex_ling_ling2, users_scores_all_isues_train])
    X_test_ngram_lex_ling_ling2_user = scipy.sparse.hstack([X_test_ngram_lex_ling_ling2, users_scores_all_isues_test])

    X_train_ngram_lex_ling_user_user2 = scipy.sparse.hstack([X_train_ngram_lex_ling, users_scores_all_isues_train, voters_religious_scores_train])
    X_test_ngram_lex_ling_user_user2 = scipy.sparse.hstack([X_test_ngram_lex_ling, users_scores_all_isues_test, voters_religious_scores_test])

    X_train_ngram_lex_ling_ling2_user_user2 = scipy.sparse.hstack([X_train_ngram_lex_ling_ling2, users_scores_all_isues_train, voters_religious_scores_train])
    X_test_ngram_lex_ling_ling2_user_user2 = scipy.sparse.hstack([X_test_ngram_lex_ling_ling2, users_scores_all_isues_test, voters_religious_scores_test])


    return(X_train_ngram_lex_ling_user, X_test_ngram_lex_ling_user, X_train_ngram_lex_ling_ling2_user, X_test_ngram_lex_ling_ling2_user, X_train_ngram_lex_ling_user_user2, X_test_ngram_lex_ling_user_user2, X_train_ngram_lex_ling_ling2_user_user2, X_test_ngram_lex_ling_ling2_user_user2)












    #####################################################
    # CODE BELOW HERE WAS JUST USED FOR TESTING IMPACT 
    # OF FEATURES ON RELIGIOUS AND NON-RELIGIOUS DEABTES 
    #####################################################

    # y_train = get_debates_labels(debates_train)


    # religious_debates_test_text = get_debates_text(debates_test[debates_test['category']=='Religion'])
    # non_religious_debates_test_text = get_debates_text(debates_test[debates_test['category']!='Religion'])  



    # y_religious_test = get_debates_labels(debates_test[debates_test['category']=='Religion'])
    # y_non_religious_test = get_debates_labels(debates_test[debates_test['category']!='Religion'])   



    # # %%
    # # Test ngram model on religious vs. non-religious topics
    # debates_train_text = get_debates_text(debates_train)
    # debates_train_text = [debate.lower() for debate in debates_train_text]

    # vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.9, stop_words='english', max_features=200, strip_accents='ascii', token_pattern=r'(?u)\b[A-Za-z]+\b')
    # X_ngrams_train = vectorizer.fit_transform(debates_train_text)

    # X_ngrams_religious_test = vectorizer.transform(religious_debates_test_text)

    # # %%
    # clf_ngrams = LogisticRegression()
    # clf_ngrams.fit(X_ngrams_train, y_train)

    # y_ngrams_religious_predicted = clf_ngrams.predict(X_ngrams_religious_test)
    # print("RELIGIOUS NGRAMS")
    # print(classification_report(y_religious_test, y_ngrams_religious_predicted))

    # # %%
    # X_ngrams_non_religious_test = vectorizer.transform(non_religious_debates_test_text)

    # # %%
    # y_ngrams_non_religious_predicted = clf_ngrams.predict(X_ngrams_non_religious_test)
    # print("NON RELIGIOUS NGRAMS")
    # print(classification_report(y_non_religious_test, y_ngrams_non_religious_predicted))




    # # Extract the word score pairs from an NRC-VAD text file
    # # that have a score greater than the given min_score param
    # # Takes in int and an nrc_vad_lexicon text file with word score pairs
    # def get_high_vad_words(min_score, lexicon):
    #     word_score_pairs = []
    #     for word in lexicon:
    #         word_score_pair = word.removesuffix('\n').split('\t')
    #         word_score_pair[1] = float(word_score_pair[1])
    #         if word_score_pair[1] > min_score:
    #             word_score_pairs.append(word_score_pair)
    #     return(word_score_pairs)


    # debates_train_text = [debate.lower() for debate in get_debates_text(debates_train)] 


    # def get_vad_words(min_score, lexicon):
    #     high_vad_words = get_high_vad_words(min_score, lexicon)
    #     tokenized_debates = [nltk.word_tokenize(debate) for debate in debates_train_text]
    #     vad_features = []
    #     for word in high_vad_words:
    #         for debate in tokenized_debates:
    #             if word[0] in debate:
    #                 vad_features.append(word[0])
    #                 break
    #     return(list(set(vad_features)))

    # # %%

    # # %%
    # # Create matrices of debates as rows and high vad words as columns
    # # with the values being the frequency of those words in each debate
    # arousal_words = get_vad_words(.97, nrc_arousal_lexicon)
    # vad_words = arousal_words 
    # vad_vectorizer = CountVectorizer(vocabulary=vad_words)

    # X_train_ngram_lex, X_test_ngram_lex = get_ngrams_lex_features(debates_train, debates_test, nrc_arousal_lexicon)

    # clf_ngrams_lex = LogisticRegression()
    # clf_ngrams_lex.fit(X_train_ngram_lex, y_train)

    # # Test ngram + lex model on religious vs. non-religious topics
    # X_religious_test_vad = vad_vectorizer.transform(religious_debates_test_text)
    # X_religious_test_ngram_lex = scipy.sparse.hstack([X_ngrams_religious_test, X_religious_test_vad])
    # y_ngrams_lex_religious_predicted = clf_ngrams_lex.predict(X_religious_test_ngram_lex)
    # print("RELIGIOUS NGRAMS+LEX")
    # print(classification_report(y_religious_test, y_ngrams_lex_religious_predicted))

    # # %%
    # X_non_religious_test_vad = vad_vectorizer.transform(non_religious_debates_test_text)
    # X_non_religious_test_ngram_lex = scipy.sparse.hstack([X_ngrams_non_religious_test, X_non_religious_test_vad])
    # y_ngrams_lex_non_religious_predicted = clf_ngrams_lex.predict(X_non_religious_test_ngram_lex)
    # print("NON RELIGIOUS NGRAMS+LEX")
    # print(classification_report(y_non_religious_test, y_ngrams_lex_non_religious_predicted))





    # def get_length_of_sides(debates):
    #     all_pro_lengths = []
    #     all_con_lengths = []
    #     for debate in debates['rounds']:
    #         pro_length = 0
    #         con_length = 0
    #         for round in debate:
    #             for side in round:
    #                 if side['side'] == "Pro":
    #                     pro_length = pro_length + len(side['text'])
    #                 if side['side'] == "Con":
    #                     con_length = con_length + len(side['text'])
    #         all_pro_lengths.append(pro_length)
    #         all_con_lengths.append(con_length)
    #     return(np.array(all_pro_lengths), np.array(all_con_lengths))


    # lengths_of_debates_religious_test = np.transpose(get_length_of_sides(debates_test[debates_test['category']=='Religion']))
    # clf_ngrams_lex_ling = LogisticRegression()
    # clf_ngrams_lex_ling.fit(X_train_ngram_lex_ling, y_train)

    # X_religious_test_ngram_lex_ling = scipy.sparse.hstack([X_religious_test_ngram_lex, lengths_of_debates_religious_test])
    # y_ngrams_lex_ling_religious_predicted = clf_ngrams_lex_ling.predict(X_religious_test_ngram_lex_ling)
    # print("RELIGIOUS NGRAMS+LEX+LING")
    # print(classification_report(y_religious_test, y_ngrams_lex_ling_religious_predicted))

    # # %%


    # lengths_of_debates_non_religious_test = np.transpose(get_length_of_sides(debates_test[debates_test['category']!='Religion']))
    # X_non_religious_test_ngram_lex_ling = scipy.sparse.hstack([X_non_religious_test_ngram_lex, lengths_of_debates_non_religious_test])
    # y_ngrams_lex_ling_religious_predicted = clf_ngrams_lex_ling.predict(X_non_religious_test_ngram_lex_ling)

    # y_ngrams_lex_ling_non_religious_predicted = clf_ngrams_lex_ling.predict(X_non_religious_test_ngram_lex_ling)
    # print("NON RELIGIOUS NGRAMS+LEX+LING")
    # print(classification_report(y_non_religious_test, y_ngrams_lex_ling_non_religious_predicted))








    # regligious_debates_tokenized = [nltk.word_tokenize(debate) for debate in religious_debates_test_text]


    # def get_prp_counts(tokenized_text):
    #     pos_tokens = nltk.pos_tag_sents(tokenized_text)
    #     prp_counts = []
    #     for debate in pos_tokens:
    #         prp_count = 0
    #         for token in debate:
    #             if token[1] == 'PRP':
    #                 prp_count = prp_count + 1
    #         prp_counts.append(prp_count)
    #     return(np.array(prp_counts))


    # prp_religious_test = get_prp_counts(regligious_debates_tokenized)
    # # %%
    # prp_religious_test_reshape = prp_religious_test.reshape(prp_religious_test.shape+(1,))

    # clf_ngrams_lex_ling_ling2 = LogisticRegression()
    # clf_ngrams_lex_ling_ling2.fit(X_train_ngram_lex_ling_ling2, y_train)

    # X_religious_test_ngram_lex_ling_ling2 = scipy.sparse.hstack([X_religious_test_ngram_lex_ling, prp_religious_test_reshape])
    # y_ngrams_lex_ling_ling2_religious_predicted = clf_ngrams_lex_ling_ling2.predict(X_religious_test_ngram_lex_ling_ling2)
    # print("RELIGIOUS NGRAMS+LEX+LING+LING2")
    # print(classification_report(y_religious_test, y_ngrams_lex_ling_ling2_religious_predicted))

    # # %%
    # non_regligious_debates_tokenized = [nltk.word_tokenize(debate) for debate in non_religious_debates_test_text]
    # #%%
    # prp_non_religious_test = get_prp_counts(non_regligious_debates_tokenized)
    # # %%
    # prp_non_religious_test_reshape = prp_non_religious_test.reshape(prp_non_religious_test.shape+(1,))

    # X_non_religious_test_ngram_lex_ling_ling2 = scipy.sparse.hstack([X_non_religious_test_ngram_lex_ling, prp_non_religious_test_reshape])
    # y_ngrams_lex_ling_ling2_non_religious_predicted = clf_ngrams_lex_ling_ling2.predict(X_non_religious_test_ngram_lex_ling_ling2)
    # print("NON RELIGIOUS NGRAMS+LEX+LING+LING2")
    # print(classification_report(y_non_religious_test, y_ngrams_lex_ling_ling2_non_religious_predicted))





    # # Test ngram + lex + ling + ling2 + user (and + user2) model on religious vs. non-religious topics
    # debates_test_religious = debates_test[debates_test['category'] == 'Religion']
    # debates_test_non_religious = debates_test[debates_test['category'] != 'Religion']
    # #%%
    # voter_counts_religious_test = get_users_scores_all_issues(debates_test_religious)
    # voter_counts_non_religious_test = get_users_scores_all_issues(debates_test_non_religious)

    # voter_counts_religious_test2 = get_voters_religious_scores(debates_test_religious)
    # voter_counts_non_religious_test2 = get_voters_religious_scores(debates_test_non_religious)

    # #%%
    # X_test_ngram_lex_ling_ling2_user_religious = scipy.sparse.hstack([X_religious_test_ngram_lex_ling_ling2, voter_counts_religious_test])
    # X_test_ngram_lex_ling_ling2_user_non_religious = scipy.sparse.hstack([X_non_religious_test_ngram_lex_ling_ling2, voter_counts_non_religious_test])


    # X_test_ngram_lex_ling_ling2_user_user2_religious = scipy.sparse.hstack([X_test_ngram_lex_ling_ling2_user_religious, voter_counts_religious_test2])
    # X_test_ngram_lex_ling_ling2_user_user2_non_religious = scipy.sparse.hstack([X_test_ngram_lex_ling_ling2_user_non_religious, voter_counts_non_religious_test2])



    # X_train_ngram_lex_ling, X_test_ngram_lex_ling, X_train_ngram_lex_ling_ling2, X_test_ngram_lex_ling_ling2 = get_ngrams_lex_ling_features(debates_train, debates_test, nrc_arousal_lexicon)
    # clf_ngrams_lex_ling_ling2 = LogisticRegression()
    # clf_ngrams_lex_ling_ling2.fit(X_train_ngram_lex_ling_ling2, y_train)

    # clf_ngrams_lex_ling_ling2_user = LogisticRegression()
    # clf_ngrams_lex_ling_ling2_user.fit(X_train_ngram_lex_ling_ling2_user, y_train)

    # clf_ngrams_lex_ling_ling2_user_user2 = LogisticRegression()
    # clf_ngrams_lex_ling_ling2_user_user2.fit(X_train_ngram_lex_ling_ling2_user_user2, y_train)



    # # %%
    # y_ngrams_lex_ling_ling2_user_religious_predicted = clf_ngrams_lex_ling_ling2_user.predict(X_test_ngram_lex_ling_ling2_user_religious)
    # print("RELIGIOUS NGRAMS+LEX+LING+LING2+USER")
    # print(classification_report(y_religious_test, y_ngrams_lex_ling_ling2_user_religious_predicted))

    # y_ngrams_lex_ling_ling2_user_non_religious_predicted = clf_ngrams_lex_ling_ling2_user.predict(X_test_ngram_lex_ling_ling2_user_non_religious)
    # print("NON RELIGIOUS NGRAMS+LEX+LING+LING2+USER")
    # print(classification_report(y_non_religious_test, y_ngrams_lex_ling_ling2_user_non_religious_predicted))


    # y_ngrams_lex_ling_ling2_user_user2_religious_predicted = clf_ngrams_lex_ling_ling2_user_user2.predict(X_test_ngram_lex_ling_ling2_user_user2_religious)
    # print("RELIGIOUS NGRAMS+LEX+LING+LING2+USER+USER2")
    # print(classification_report(y_religious_test, y_ngrams_lex_ling_ling2_user_user2_religious_predicted))

    # y_ngrams_lex_ling_ling2_user_user2_non_religious_predicted = clf_ngrams_lex_ling_ling2_user_user2.predict(X_test_ngram_lex_ling_ling2_user_user2_non_religious)
    # print("NON RELIGIOUS NGRAMS+LEX+LING+LING2+USER+USER2")
    # print(classification_report(y_non_religious_test, y_ngrams_lex_ling_ling2_user_user2_non_religious_predicted))

